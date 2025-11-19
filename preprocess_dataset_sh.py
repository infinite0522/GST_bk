from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
import cv2
import argparse


def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w * ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w * ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h * ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h * ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def find_dis(point):
    square = np.sum(point * points, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2 * np.matmul(point, point.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis


def generate_data(im_path):
    im = Image.open(im_path)
    im_w, im_h = im.size
    # 获取标注点坐标
    mat_path = im_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_')
    points = loadmat(mat_path)['image_info'][0, 0][0, 0][0]
    # print(points)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    # 图像缩放（放大），返回缩放后的图片宽高和缩放比例
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    # 根据缩放比例处理点坐标信息
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr

    os.remove(mat_path)
    return Image.fromarray(im), points


def Imgpath(path):
    # 获取图片文件地址
    files = os.listdir(path)
    img_paths = []
    for filename in files:
        portion = os.path.splitext(filename)
        print(portion[0][-7:])
        if (portion[1] == '.jpg') & (portion[0][:-7] != 'entropy'):
            img_paths.append(os.path.join(path, filename))
    return img_paths


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--origin-dir', default='/home/ubuntu/datasets/Counting/Shanghai/part_A_final',
                        help='original data directory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    min_size = 512
    max_size = 2048
    for phase in ['Train', 'Test']:
        # 处理训练集和验证集
        if phase == 'Train':
            sub_phase_list = ['train', 'val']
            for sub_phase in sub_phase_list:
                sub_save_dir = os.path.join(args.origin_dir, sub_phase)
                print(sub_save_dir)
                # 获取图片文件地址
                img_paths = Imgpath(sub_save_dir)
                print(img_paths)
                # 遍历图片文件地址
                for im_path in img_paths:
                    # 获取图片名
                    name = os.path.basename(im_path)
                    print(sub_phase + ": " + name)
                    # 生成图片文件im和点坐标points
                    im, points = generate_data(im_path)
                    # 如果是训练集则做额外的处理
                    if sub_phase == 'train':
                        dis = find_dis(points)
                        print(dis)
                        points = np.concatenate((points, dis), axis=1)
                    # 保存缩放后的图片和点坐标
                    im_save_path = os.path.join(sub_save_dir, name)
                    im.save(im_save_path)
                    gd_save_path = im_save_path.replace('jpg', 'npy')
                    np.save(gd_save_path, points)
        # 处理测试集
        else:
            sub_save_dir = os.path.join(args.origin_dir, 'test')
            img_paths = Imgpath(sub_save_dir)
            for im_path in img_paths:
                name = os.path.basename(im_path)
                print("test: " + name)
                # print(name)
                im, points = generate_data(im_path)
                im_save_path = os.path.join(sub_save_dir, name)
                im.save(im_save_path)
                gd_save_path = im_save_path.replace('jpg', 'npy')
                np.save(gd_save_path, points)

