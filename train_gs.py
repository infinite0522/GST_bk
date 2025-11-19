# from utils.regression_trainer_multigs import RegTrainer
from utils.regression_trainer_gs import RegTrainer
from utils.helper import setup_seed
import argparse
import os
import torch
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--data-dir', default='/home/ubuntu/datasets/Counting/UCF-Train-Val-Test',   #'/home/ubuntu/datasets/Counting/Shanghai/part_A_train-val-test',
                        help='training data directory')
    parser.add_argument('--save-dir', default='./saved_models/2dgs_1gs_noshape/qnrf',  #'./saved_models/2dgs_multigs/ShanghaiA',
                        help='directory to save models.')

    parser.add_argument('--lr', type=float, default=1e-5,   # default=1e-5, shanghaia :1e-6
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,     # default=1e-4
                        help='the weight decay')
    parser.add_argument('--milestones', type=str, default='1')
    parser.add_argument('--gamma', type=float, default=1,
                        help='gamma')
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=1000,  # default=1000,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=5,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=50,   # default=600
                        help='the epoch start to val')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')

    parser.add_argument('--is-gray', type=bool, default=False,
                        help='whether the input image is gray')
    parser.add_argument('--crop-size', type=int, default=512,   # vit:384 512
                        help='the crop size of the train image')
    parser.add_argument('--downsample-ratio', type=int, default=8,
                        help='downsample ratio')

    parser.add_argument('--use-background', type=bool, default=True,    # default:False
                        help='whether to use background modelling')
    parser.add_argument('--sigma', type=float, default=8.0,
                        help='sigma for likelihood')
    parser.add_argument('--background-ratio', type=float, default=0.015,  # default:0.15
                        help='background ratio')
    parser.add_argument('--post-min', type=float, default=2.0,
                        help='post min')
    parser.add_argument('--post-max', type=float, default=30.0,
                        help='post max')
    parser.add_argument('--scale-ratio', type=float, default=1.0,
                        help='scale ratio')
    parser.add_argument('--cut-off', type=float, default=3.0,
                        help='cut off')
    parser.add_argument('--sigma1', type=float, default=0.08,       # > 0.08 or at least > 0.05
                        help='spatial sigma for gs similarity computing')
    parser.add_argument('--sigma2', type=float, default=0.5,
                        help='color sigma for gs similarity computing')
    parser.add_argument('--wbay', type=float, default=1.0,
                        help='weight for bayesian loss')
    parser.add_argument('--wco', type=float, default=1.0,
                        help='weight for count loss')

    parser.add_argument('--mode', type=str, default='mul_add',
                        help='mul or 1 or 1_add')
    parser.add_argument('--scale-standard', type=str, default='',
                        help='The approach to standardization the scales')
    #parser.add_argument('--scale-standard-vec', type=bool, default=False,
    #                    help='whether to standardization the scales')
    #parser.add_argument('--scale-round', type=bool, default=False,
    #                    help='whether to standardization the scales')
    #parser.add_argument('--upper', type=int, default=20,
    #                    help='whether to standardization the scales')
    #parser.add_argument('--bin-num', type=int, default=10,
    #                    help='whether to standardization the scales')


    parser.add_argument('--h-ratio', type=float, default=6.0,  #
                        help='gt_2dgs_scale = gt_h / h_ratio, only for 2dgs_disgt experiment on JHU++')

    parser.add_argument('--seed', type=int, default=4073,
                        help='mul or 1 or 1_add')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    #setup_seed(4073)
    #seed_list = [3407, 81375]
    #seed_list = [int(e) for e in args.seeds.split(',')]
    args = parse_args()
    setup_seed(args.seed)

    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()
