import copy
import torch
import os.path
import argparse
import alphashape
import numpy as np
from model import Pct
from tqdm import tqdm
import torch.nn as nn
import networkx as nx
from primitive import *
from osgeo import gdal, osr
from reconstruct_utils import *
import torch.nn.functional as F
from bldg_regularization import *
from skspatial.objects import Plane
from data import Clouds, get_rotation
from shapely.ops import cascaded_union
from torch.utils.data import DataLoader
from shapely.geometry import MultiPoint
from scipy.spatial.distance import cdist
from trimesh.creation import triangulate_polygon
from scipy.optimize import differential_evolution


np.bool = np.bool_
os.environ['PATH'] += os.pathsep + 'C:\\Program Files\\CloudCompare'

parser = argparse.ArgumentParser(description='Point Cloud Recognition')
parser.add_argument('--exp_name', type=str, default='roof_primitive', metavar='N',
                    help='Name of the experiment')
parser.add_argument('--dataset', type=str, default='roof_primitive', metavar='N')
parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of episode to train ')
parser.add_argument('--use_sgd', type=bool, default=False,
                    help='Use SGD')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--eval', type=bool, default=False,
                    help='evaluate the model')
parser.add_argument('--num_points', type=int, default=1024,
                    help='num of points to use')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout rate')
parser.add_argument('--model_path', type=str, default='', metavar='N',
                    help='Pretrained model path')
parser.add_argument('--clear_log', type=bool, default=True,
                    help='Clear log rile')
args = parser.parse_args()

with open('primitive_info.json') as f:
    primitive_dict = json.load(f)
    roof_types = list(primitive_dict.keys())
    dim_num = [primitive_dict[roof_type]['para_num'] + 1 for roof_type in roof_types]
    roof_func = [globals()[primitive_dict[roof_type]['func_name']] for roof_type in roof_types]

device = torch.device("cuda")
model_cls_fp = f'checkpoints/roof_primitive/models/model_cls.t7'
model_cls = nn.DataParallel(Pct(args, len(dim_num)).to(device))
model_cls.load_state_dict(torch.load(model_cls_fp))
model_cls.eval()

model_reg = []
for idx, dim in enumerate(dim_num):
    model_reg_fp = f'checkpoints/roof_primitive/models/model_reg_{roof_types[idx]}.t7'
    model = nn.DataParallel(Pct(args, dim).to(device))
    model.load_state_dict(torch.load(model_reg_fp))
    model.eval()
    model_reg.append(model)


def estimate_candidate_dl(candidates, model_cls, model_reg):
    loader = DataLoader(Clouds(candidates), batch_size=16, shuffle=False, drop_last=False)
    samples_all, class_all = [], []
    scales_all, translations_all, paras_all = [], [], []
    verts_all, faces_all, bbx_all = [], [], []
    idx, scores, sources = 0, [], []
    verts_src_all, bbx_src_all, sample_src_all, rads_all = [], [], [], []

    for _, data in enumerate(tqdm(loader, desc='    Evaluating candidates...')):
        item, scales, translations = data
        item = item.to(device)

        clouds = item.detach().cpu().numpy()
        logits_cls = model_cls(item.permute(0, 2, 1))
        pred_probs = F.softmax(logits_cls, dim=1)
        classes = pred_probs.max(dim=1)[1].detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        translations = translations.detach().cpu().numpy()

        for i in range(len(clouds)):
            cls, scale, translation = classes[i], scales[i], translations[i]
            sources.append(candidates[idx])
            class_all.append(roof_types[cls])
            scales_all.append(scale)
            translations_all.append(translation)
            logits = model_reg[cls](item[None, i].permute(0, 2, 1)).detach().cpu().numpy()[0]
            rad = get_rotation(primitive_dict[roof_types[cls]]['cycle'], logits[0])

            if primitive_dict[roof_types[cls]]['para_num'] == 3:
                para = [logits[1], logits[2]*1.1, 1, 1, logits[3]]
                bbx = bounding_box([logits[1], logits[2], logits[3]])
            elif primitive_dict[roof_types[cls]]['para_num']  == 4:
                para = [logits[1], logits[2]*1.1, 1, logits[3], logits[4]]
                bbx = bounding_box([logits[1], logits[2], logits[4]])
            else:
                para = [logits[1], logits[2]*1.1, logits[3], logits[4], logits[5]]
                bbx = bounding_box([logits[1], logits[2], logits[5]])

            verts_src, faces, _ = roof_func[cls](para, True, False)
            verts_src_all.append(verts_src)
            bbx_src_all.append(bbx)
            sample_src_all.append(clouds[i])
            rads_all.append(rad)
            verts, bbx = pc_RT(verts_src, rad), pc_RT(bbx, rad)

            DTM, MTD = get_reconstruction_score(verts, faces, clouds[i])
            scores.append(DTM + MTD)
            verts_all.append((verts + translation) * scale)
            samples_all.append((clouds[i] + translation) * scale)
            bbx_all.append((bbx + translation) * scale)
            faces_all.append(faces)
            paras_all.append(para)

            idx += 1
    bid = np.argmin(scores)

    # for bid in np.argsort(scores)[:10]:
    #     vis_primitive(class_all[bid], verts_all[bid], faces_all[bid], samples_all[bid], scores[bid])
    rst = differential_evolution(obj, [(- np.pi, np.pi)], x0=rads_all[bid],
                                 args=([verts_src_all[bid], faces_all[bid], sample_src_all[bid]],))
    DTM, MTD = get_reconstruction_score(pc_RT(verts_src_all[bid], rst.x[0]), faces_all[bid], sample_src_all[bid])
    score = DTM + MTD
    verts = (pc_RT(verts_src_all[bid], rst.x[0]) + translations_all[bid]) * scales_all[bid]
    bbx = (pc_RT(bbx_src_all[bid], rst.x[0]) + translations_all[bid]) * scales_all[bid]

    return (bid, score, verts, faces_all[bid], bbx, class_all[bid],
            samples_all[bid], paras_all[bid], scales_all[bid], translations_all[bid])


def check_multipolygon(poly, type_return='convex'):
    if poly.geom_type == 'MultiPolygon':
        geoms = [geom for geom in poly.geoms]
        areas = [geom.area for geom in poly.geoms]
        if type_return == 'max':
            out_geom = [geoms[np.argmax(areas)]]
        elif type_return == 'all':
            out_geom = geoms
        else:
            out_geom = [poly.convex_hull]
    elif poly.geom_type == 'Polygon':
        out_geom = [poly]
    else:
        out_geom = [poly.convex_hull]
    return out_geom


def reconstruct_planes(planes, APD, dem_plane):
    DTM, MTD, IOU, meshes, roof_meshes = [], [], [], [], []
    planes_para_all = [Plane.best_fit(item) for item in planes]
    threshold = 1.5 * APD
    connectivity_G = nx.from_numpy_array(get_connectivity(planes, threshold))
    planes_cluster = [list(connectivity_G.subgraph(c).nodes()) for c in nx.connected_components(connectivity_G)]

    for nodes in tqdm(planes_cluster, desc=f'Reconstructing, groups: {len(planes_cluster)}...'):
        clouds = np.concatenate([planes[i] for i in nodes])
        border_ashape = check_multipolygon(alphashape.alphashape(clouds[:, 0:2], .2).simplify(APD))
        border_reg, direction = regularize(border_ashape)
        border_reg = border_reg[0]

        dists = np.concatenate([planes_para_all[idx].distance_points(planes[idx]) for idx in nodes])
        DTM.append(np.sqrt(np.mean(np.square(dists))))
        IOU.append(border_reg.intersection(border_ashape[0]).area / border_reg.union(border_ashape[0]).area)
        MTD.append(np.mean(np.square(cdist(np.array(border_reg.exterior.coords[:]), clouds[:, 0:2]).min(axis=1))))

        planes_shape, planes_para = [], []
        for idx in nodes:
            shapes = check_multipolygon(alphashape.alphashape(planes[idx][:, 0:2], .2), 'all')

            for shape in shapes:
                planes_shape.append(shape.simplify(APD).buffer(APD / 2, join_style='mitre'))
                planes_para.append(planes_para_all[idx])

        planes_reg = [check_multipolygon(border_reg.intersection(poly), 'max')[0]
                      for poly in regularize(planes_shape, False, 2, direction)]
        ids = [item for item in range(len(planes_reg))]

        for i in ids:
            for j in [item for item in ids if item != i]:
                if planes_reg[i].intersects(planes_reg[j]):
                    if planes_reg[i].area > planes_reg[j].area:
                        diff = planes_reg[j].difference(planes_reg[i])
                        planes_reg[j] = check_multipolygon(diff, type_return='max')[0]
                    else:
                        diff = planes_reg[i].difference(planes_reg[j])
                        planes_reg[i] = check_multipolygon(diff, type_return='max')[0]
        planes_para = [planes_para[i] for i in ids if not planes_reg[i].is_empty and
                       planes_reg[i].geom_type != 'LineString']
        planes_reg = [planes_reg[i] for i in ids if not planes_reg[i].is_empty and
                      planes_reg[i].geom_type != 'LineString']

        remaining = border_reg.difference(cascaded_union(planes_reg))
        if not remaining.is_empty:
            remaining_temp = check_multipolygon(remaining, type_return='all')
            remaining = []
            for poly in remaining_temp:
                flag, rst = simplify_by_angle(poly)
                if flag:
                    remaining.append(rst)
            remaining_para = []

            for poly in remaining:
                poly_np = np.asarray(poly.exterior.coords[:])
                dists = [cdist(poly_np, np.asarray(item.exterior.coords[:])).min(axis=1).mean() for item in planes_reg]
                remaining_para.append(planes_para[np.argmin(dists)])

            planes_reg = planes_reg + remaining
            planes_para = planes_para + remaining_para

        mesh_all = []

        for i in range(len(planes_reg)):
            if abs(np.degrees(np.arccos(planes_para[i].normal @ [0, 0, 1])) - 90) > 5:
                vertices, faces = triangulate_polygon(planes_reg[i])
                vertices = np.array([get_vert_z(p, planes_para[i].point, planes_para[i].normal) for p in vertices])

                if len(faces) == 0:
                    faces = [[3, 0, 1, 2]]
                else:
                    faces = np.insert(faces, 0, 3, axis=1)

                roof_mesh = pv.PolyData(vertices, faces)
                roof_meshes.append(roof_mesh)
                mesh = roof_mesh.extrude_trim((0, 0, -1.0), dem_plane)
                mesh_all.append(mesh)

        meshes.append(mesh_all)
    return DTM, MTD, IOU, meshes


def get_bldg_xyzs(bldg_ext, dem, ct, gt_inv):
    bldg_ext_xyz = []
    for pt in bldg_ext:
        mapx, mapy, z = ct.TransformPoint(pt[0], pt[1])
        pixel_x, pixel_y = gdal.ApplyGeoTransform(gt_inv, mapx, mapy)
        pixel_x = round(pixel_x)
        pixel_y = round(pixel_y)
        pixel_x = max(min(pixel_x, width - 1), 0)
        pixel_y = max(min(pixel_y, height - 1), 0)
        bldg_ext_xyz.append([pt[0], pt[1], dem[pixel_y, pixel_x] * 3.28083333])
    return np.array(bldg_ext_xyz)


if __name__ == "__main__":
    np.random.seed(42)
    pc_fd = 'data/reconstruction/cloud'
    out_fd = 'data/reconstruction/meshes'
    if os.path.exists(out_fd):
        shutil.rmtree(out_fd)
    os.mkdir(out_fd)

    pc_fps = [os.path.join(pc_fd, item) for item in os.listdir(pc_fd)]
    print('Loading DEM...')
    dem_fp = 'data/reconstruction/USGS_1M_16_x50y448_IN_Indiana_Statewide_LiDAR_2017_B17.tif'
    ds = gdal.Open(dem_fp)
    gt = ds.GetGeoTransform()
    dem = np.array(ds.ReadAsArray())
    width = ds.RasterXSize
    height = ds.RasterYSize
    point_srs = osr.SpatialReference()
    point_srs.ImportFromEPSG(2968)
    point_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dem_srs = osr.SpatialReference()
    dem_srs.ImportFromWkt(ds.GetProjection())
    ct = osr.CoordinateTransformation(point_srs, dem_srs)
    gt_inv = gdal.InvGeoTransform(gt)

    APD = 2.33
    batch_size, vis = 16, True

    for fp in pc_fps:
        pc_raw, t = read_las(fp)
        bldg_ext = np.array(MultiPoint(pc_raw[:, 0:2]).convex_hull.exterior.coords[:]) + t[0:2]
        bldg_ext_xyz = get_bldg_xyzs(bldg_ext, dem, ct, gt_inv) - t
        plane_coef = Plane.best_fit(bldg_ext_xyz)

        iter_num, geom_verts, geom_faces, samples, scores, plane_geoms, roofs = 1, [], [], [], [], [], []
        primitive_DTM, primitive_IOU, primitive_MTD, primitive_mesh = [], [], [], []
        building_DTM, building_IOU, building_MTD = [], [], []

        smin, smax = pc_raw.min(axis=0), pc_raw.max(axis=0)
        dem_plane = pv.Plane(center=tuple(plane_coef.point), direction=tuple(plane_coef.normal),
                             i_size=int(smax[0] - smin[0] + 500), j_size=int(smax[1] - smin[1] + 500))
        height = plane_coef.distance_points(pc_raw)
        pc_raw = pc_raw[height >= 3.28 * 2]

        # pl = pv.Plotter()
        # pl.add_points(pv.PolyData(pc_raw), render_points_as_spheres=True, point_size=5, color='red')
        # pl.show()

        planes = ransac_cc(pc_raw)
        if len(planes) == 0:
            planes = ransac_o3d(pc_raw)

        # pl = pv.Plotter()
        # for cloud in planes:
        #     c = np.random.uniform(0, 1, 3)
        #     pl.add_points(pv.PolyData(cloud), render_points_as_spheres=True, point_size=5, color=c, opacity=0.6)
        # pl.show()

        planes_src = copy.deepcopy(planes)
        planes_vis = copy.deepcopy(planes)
        planes_sample = smaple_planes(planes)

        while len(planes) > 0:
            print(f'Iteration {iter_num}')
            iter_num += 1
            candidates, proposals = create_candidate(planes, planes_sample, 1.5 * APD)

            if len(candidates) == 0:
                plane_DTM, plane_MTD, plane_IOU, plane_geoms = reconstruct_planes(planes_src, APD, dem_plane)
                break
            else:
                print(f'    Detected candidates: {len(candidates)}')
                rst = estimate_candidate_dl(candidates, model_cls, model_reg)
                bid, score, verts, faces, bbx, cls, sample, para, translation, scale = rst

                if score < 0.08:
                    geom_DTM = get_DTM(verts, faces, sample)
                    geom_MDT = np.sqrt(np.mean(cdist(verts, sample).min(axis=1) ** 2))
                    data_poly = MultiPoint(sample[:, 0:2]).convex_hull
                    model_poly = MultiPoint(verts[:, 0:2]).convex_hull
                    geom_IOU = data_poly.intersection(model_poly).area / data_poly.union(model_poly).area
                    planes, planes_src, planes_sample = get_outliers(planes, planes_src, planes_sample,
                                                                     verts, faces, proposals[bid], geom_DTM)
                    geom_verts.append(verts)
                    geom_faces.append(faces)
                    samples.append(sample)
                    scores.append(score)

                    primitive_mesh.append(pv.PolyData(verts, np.insert(faces, 0, 3, axis=1)).
                                          extrude_trim((0, 0, -1.0), dem_plane))

                    primitive_DTM.append(geom_DTM)
                    primitive_MTD.append(geom_MDT)
                    primitive_IOU.append(geom_IOU)
                    roofs.append(cls)
                else:
                    plane_DTM, plane_MTD, plane_IOU, plane_geoms = reconstruct_planes(planes_src, APD, dem_plane)
                    break

        fname = os.path.basename(fp).split('.')[0]

        for idx, mesh in enumerate(primitive_mesh):
            out_fp = os.path.join(out_fd, f'{fname}_{roofs[idx]}_{idx}.obj')
            mesh.translate(t).save(out_fp)

        for idx, plane_set in enumerate(plane_geoms):
            for jdx, mesh in enumerate(plane_set):
                out_fp = os.path.join(out_fd, f'{fname}_plane_{idx}_{jdx}.obj')
                mesh.translate(t).save(out_fp)

        pl = pv.Plotter()
        for cloud in planes_vis:
            c = np.random.uniform(0, 1, 3)
            pl.add_points(pv.PolyData(cloud), render_points_as_spheres=True, point_size=5, color=c)
        for mesh in primitive_mesh:
            pl.add_mesh(mesh, color='green', show_edges=True)
        for plane_set in plane_geoms:
            for mesh in plane_set:
                pl.add_mesh(mesh, show_edges=True)
        pl.show()
        # path = pl.generate_orbital_path(factor=2.0, n_points=72, viewup=[0, 0, 2], shift=1.5 * (smax[2] - smin[2]))
        # pl.open_gif(f'gifs\\{os.path.basename(fp)}.gif')
        # pl.orbit_on_path(path, write_frames=True, viewup=[0, 0, 1], step=0.05)
        # pl.close()
        c = 1

        # vis_primitive(cls, verts, faces, sample, score)
        # vis_primitives(geom_verts, geom_faces, plane_geoms, pc_raw)

