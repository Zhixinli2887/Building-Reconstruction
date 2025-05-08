from __future__ import print_function
import torch
import argparse
import warnings
import numpy as np
import torch.nn as nn
from model import Pct_reg
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='Point Cloud Recognition')
parser.add_argument('--exp_name', type=str, default='roof_primitive_perfect', metavar='N',
                    help='Name of the experiment')
parser.add_argument('--dataset', type=str, default='roof_primitive_perfect', metavar='N')
parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                    help='number of episode to train ')
parser.add_argument('--use_sgd', type=bool, default=False,
                    help='Use SGD')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=3407, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--eval', type=bool, default=False,
                    help='evaluate the model')
parser.add_argument('--num_points', type=int, default=512,
                    help='num of points to use')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout rate')
parser.add_argument('--model_path', type=str, default='', metavar='N',
                    help='Pretrained model path')
parser.add_argument('--clear_log', type=bool, default=True,
                    help='Clear log rile')
args = parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_reg = nn.DataParallel(Pct_reg(args, 4).to(device))

    opt_old = optim.AdamW(model_reg.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_new = optim.AdamW(model_reg.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler_old = CosineAnnealingLR(opt_old, 2000, eta_min=args.lr)
    scheduler_new = CosineAnnealingLR(opt_new, 2000, eta_min=args.lr*0.01)

    learning_rate_old, learning_rate_new = [], []

    for epoch in range(2000):
        scheduler_old.step()
        learning_rate_old.append(scheduler_old.get_lr()[0])

    for epoch in range(2000):
        scheduler_new.step()
        learning_rate_new.append(scheduler_new.get_lr()[0])

    learning_rate_old, learning_rate_new = np.array(learning_rate_old), np.array(learning_rate_new)
    fig, ax = plt.subplots()
    ax.plot(learning_rate_old, label='old')
    ax.plot(learning_rate_new, label='new')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    ax.set_title('Comparison of Learning Rates')
    ax.legend()
    plt.ylim(0, 0.00011)
    plt.show()
    c = 1

