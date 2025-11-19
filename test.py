import cv2
import torch
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt, cm
#from datasets.crowd import Crowd
from datasets.crowd_ucf_1gs import Crowd
#from datasets.crowd_gs_sh import Crowd
from models.vgg import vgg19
from models.vit import ViT_c
import argparse
import math

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='/home/ubuntu/datasets/Counting/UCF-Train-Val-Test',  #Shanghai/part_B_train-val-test',
                        help='training data directory')
    parser.add_argument('--save-dir', default='saved_models/2dgs_1gs/qnrf', #'./saved_models/2dgs_multigs/1007-012440',
                        help='model directory')
    parser.add_argument('--dir-date', default='1212-103301',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)
    model = vgg19()
    # model = ViT_c()
    device = torch.device('cuda')
    model.to(device)
    save_dir = os.path.join(args.save_dir, args.dir_date)
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth'), device))
    print(os.path.join(args.save_dir, 'best_model.pth'))
    epoch_minus = []

    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        b, c, h, w = inputs.shape
        h, w = int(h), int(w)
        assert b == 1, 'the batch size should equal to 1 in validation mode'


        """
        input_list = []
        # 1792
        if h >= 512 or w >= 512:
            h_stride = int(math.ceil(1.0 * h / 512))
            w_stride = int(math.ceil(1.0 * w / 512))
            h_step = h // h_stride
            w_step = w // w_stride
            for i in range(h_stride):
                for j in range(w_stride):
                    h_start = i * h_step
                    if i != h_stride - 1:
                        h_end = (i + 1) * h_step
                    else:
                        h_end = h
                    w_start = j * w_step
                    if j != w_stride - 1:
                        w_end = (j + 1) * w_step
                    else:
                        w_end = w
                    input_list.append(inputs[:, :, h_start:h_end, w_start:w_end])
            with torch.set_grad_enabled(False):
                pre_count = 0.0
                for idx, input in enumerate(input_list):
                    output = model(input)[0]
                    pre_count += torch.sum(output)
            res = count[0].item() - pre_count.item()
            epoch_minus.append(res)
        else:
            with torch.set_grad_enabled(False):
                outputs = model(inputs)[0]
                res = count[0].item() - torch.sum(outputs).item()
                print(name, res, count[0].item(), torch.sum(outputs).item())
                epoch_minus.append(res)
        """
        with torch.set_grad_enabled(False):
            outputs = model(inputs)[0]
            res = count[0].item() - torch.sum(outputs).item()
            print(name, res, count[0].item(), torch.sum(outputs).item())
            epoch_minus.append(res)

        """
        可视化密度图："""
        if res > 500:
            #dm = outputs.squeeze().detach().cpu().numpy()
            #dm = np.array(np.squeeze(plt.cm.jet(dm)[:, :, :3]))
            #img = inputs.squeeze(0).cpu().detach().numpy()
            #img = np.transpose(img, (1, 2, 0))
            #img = cv2.resize(img, dm.shape[:2][::-1])
            #dm = 0.9 * dm + 0.1 * img
            #plt.imshow(dm)
            #plt.show()
            #plt.savefig(save_path)

            # Visualize density map
            outputs = outputs.squeeze().detach().cpu().numpy()

            # Normalize density map between 0 and 1
            normalized_dm = (outputs - np.min(outputs)) / (np.max(outputs) - np.min(
                outputs))  # (density_map - np.min(density_map)) / (np.max(density_map) - np.min(density_map))

            # Overlay density map on the original image
            #dm = cm.jet(normalized_density_map)
            #dm = Image.fromarray((dm * 255).astype(np.uint8))
            dm = np.array(np.squeeze(plt.cm.jet(normalized_dm)[:, :, :3]))
            img = inputs.squeeze(0).cpu().detach().numpy()
            img = np.transpose(img, (1, 2, 0))
            dm = cv2.resize(dm, img.shape[:2][::-1])
            #print(type(dm),type(img))

            # Create a composite image by blending the original image and the overlay
            composite_image = 0.9 * dm + 0.1 * img
            composite_image = np.array(composite_image)
            # print(composite_image.size)

            # ground_truth
            #for points in keypoints:
            #    x = int(points[0])
            #    y = int(points[1])
            #    cv2.circle(composite_image, (x, y), 3, (0, 255, 0), -1)

            # show estimation
            gt_count = " gt_count: " + "{:.2f}".format(count[0].item())
            pre_count = "pre_count: " + "{:.2f}".format(np.sum(outputs).item())
            # print(composite_image.shape)
            W, H = composite_image.shape[1], composite_image.shape[0]
            cv2.putText(composite_image, gt_count, (W - 775, H - 270), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
            cv2.putText(composite_image, pre_count, (W - 775, H - 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),
                        5)

            plt.imshow(composite_image)
            plt.show()



    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)
