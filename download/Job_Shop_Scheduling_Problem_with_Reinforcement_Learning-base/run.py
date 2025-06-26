#主要入口，支持命令行参数设定（如学习率 lr，折扣因子 gamma，种子 seed，步数 num-steps 等）
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:11:38 2020

@author: lvjf

run.py with args
"""

from __future__ import print_function

import argparse
from trainer import train


parser = argparse.ArgumentParser(description='JSSPRL')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--gamma', type=float, default=0.9,
                    help='discount factor for rewards (default: 0.9)')
parser.add_argument('--gae-lambda', type=float, default=1.00,
                    help='lambda parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=3,
                    help='random seed (default: 3)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1e5,
                    help='maximum length of an episode (default: 1e5)')
parser.add_argument('--episode', type=int, default=10,
                    help='How many episode to train the RL algorithm')

if __name__ == '__main__':
    
    args = parser.parse_args()
    print('start training...')
    train(args)
    