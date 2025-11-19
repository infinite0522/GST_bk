import csv
import pickle

import cv2
import yaml
from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from PIL.Image import Resampling
from matplotlib import pyplot as plt
from torchvision import transforms
import random
import numpy as np
from tqdm import tqdm
import h5py

from GS2D.image_fitting_bk2dgs_1gs import SimpleTrainer
# from gaussian_splatting_2D import GaussianSplatting2dOptimizer
from GS2D.give_required_data import coords_normalize, coords_reverse
#from losses.post_prob_gs_bk_1 import Post_Prob_GS

def image_to_tensor(img: Image, width=None, height=None):
    import torchvision.transforms as transforms

    if width is not None and height is not None:
        # transform = transforms.ToTensor()
        transform = transforms.Compose([
            transforms.Resize([width, height]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor

class Counting_2dgs:
    def __init__(self, config_file_path, is_gray=False):
        with open(config_file_path, 'r') as config_file:
            config = yaml.safe_load(config_file)

        self.root_path = config["root_path"]
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        self.im_list = [img_path for img_path in self.im_list if 'gs' not in img_path]
        self.d_ratio = config["downsample_ratio"]
        # self.KERNEL_SIZE = config["KERNEL_SIZE"]
        self.num_epochs = config["num_epochs"]
        # self.densification_interval = config["densification_interval"]
        self.use_bk_mask = config["use_bk_mask"]
        self.learning_rate = config["learning_rate"]
        self.num_back_points = config["bk_num"]
        self.gt_points = config["gt_points"]    # True or False, whether gt_points exist
        self.scale_bound = config["scale_bound"]

        #self.image_size = None
        #self.image_file_name = config["image_file_name"]
        #self.num_gt_points = None
        #self.num_back_points = config["bk_points"]
        #self.display_interval = config["display_interval"]
        #self.grad_threshold = config["gradient_threshold"]
        #self.gauss_threshold = config["gaussian_threshold"]
        #self.display_loss = config["display_loss"]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def batch_saving(self):
        output_csv_path = os.path.join(self.root_path, "gs_params_2/loss_results.csv")
        out_num = 0

        # Open a CSV file for writing the results
        with open(output_csv_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['File Name', 'Render_Loss', 'Shape_Loss'])

            for img_path in tqdm(self.im_list):

                out_img_save_path = img_path.replace('.jpg', '_gs_render.jpg').replace('IMG', 'render_2/IMG')
                gs_params_save_path = img_path.replace('.jpg', '_gs_params.h5').replace('IMG', 'gs_params_2/IMG')
                if os.path.exists(gs_params_save_path) and os.path.exists(out_img_save_path):
                    continue

                img = Image.open(img_path).convert('RGB')
                print(img.size)
                width, height = img.size
                if self.d_ratio != 1:
                    re_width, re_height = img.size[0] // self.d_ratio, img.size[1] // self.d_ratio
                    img.resize((re_width, re_height), Image.BILINEAR)
                    gt_img_tensor = image_to_tensor(img, width, height)
                else:
                    gt_img_tensor = image_to_tensor(img)

                gt_path = img_path.replace('jpg', 'npy')
                gt_points = torch.tensor(np.load(gt_path))

                if not len(gt_points) > 0:
                    plt.imsave(out_img_save_path, img)
                    params = np.array([])
                    with h5py.File(os.path.join(gs_params_save_path), 'w') as f:
                        f.create_dataset('params', data=params, compression='gzip')
                    continue

                if self.d_ratio != 1:
                    gt_points[:,:2] = coords_reverse(coords_normalize(gt_points[:,:2], [height, width]),[re_height, re_width], device=torch.device('cpu'))
                num_points = gt_points.shape[0] + self.num_back_points
                print(f"{img_path} processsing:")

                trainer = SimpleTrainer(gt_image=gt_img_tensor, num_points=num_points, gt_points=gt_points, bk_num=self.num_back_points, scale_bound=self.scale_bound, use_bk_mask=self.use_bk_mask)
                render_loss, shape_loss, out_image, means, scales, rotations, colors = trainer.train(
                    iterations=self.num_epochs,
                    lr=self.learning_rate,
                    save_imgs=False
                )

                """
                # entropy_img
                stride = 1
                post_prob = Post_Prob_GS(c_size=[height, width], stride=stride, background_ratio=1.0, use_background=True,
                                            post_min=1.41,
                                            post_max=15.0, scale_ratio=1.0,
                                            device=self.device, cut_off=3.0)
    
                prob = post_prob([scales], [rotations],
                                 [means], [min(height // stride, width // stride)])
                prob = prob[0].view(-1, height // stride, width // stride) + 1e-10
                entropy = - prob * torch.log(prob)
                entropy_img = entropy.sum(dim=0)
                entropy_img = entropy_img.cpu().detach().numpy()
                entropy_img = np.squeeze(plt.cm.jet(entropy_img)[:, :, :3])
                rgb_img = np.array(
                    img.resize((width // stride, height // stride), Resampling.BILINEAR)) / 255.
                entropy_img = 0.6 * entropy_img + 0.4 * rgb_img
    
                for point in gt_points[:,:2]:
                    x, y = point
                    cv2.circle(entropy_img, (int(x), int(y)), 3, (0, 1, 0), -1)
    
                plt.imshow(entropy_img)
                plt.show()
                
    
                ga_img_save_path = img_path.replace('.jpg', '_gs_entropy.jpg').replace('img', 'entropy/img')
                plt.imsave(ga_img_save_path, entropy_img)
                """
                if not self.scale_bound:
                    out_img_save_path = img_path.replace('.jpg', '_gs_render.jpg').replace('IMG', 'render_2/IMG')
                else:
                    out_img_save_path = img_path.replace('.jpg', '_gs_render.jpg').replace('IMG', 'render_bound/IMG')
                plt.imsave(out_img_save_path, out_image)

                if not self.scale_bound:
                    gs_params_save_path = img_path.replace('.jpg', '_gs_params.h5').replace('IMG', 'gs_params_2/IMG')
                else:
                    gs_params_save_path = img_path.replace('.jpg', '_gs_params.h5').replace('IMG', 'gs_params_bound/IMG')
                print(scales.shape, scales.shape,rotations.shape, colors.shape)
                params = torch.cat((means, scales, rotations, colors), dim=1)  # [:,:2], [:,2:4], [:,4:5], [:,5:]
                print(params.shape)
                params = params.cpu().detach().numpy()


                with h5py.File(os.path.join(gs_params_save_path), 'w') as f:
                    f.create_dataset('params', data=params, compression='gzip')

                csv_writer.writerow([os.path.basename(img_path), render_loss.cpu().detach().numpy(), shape_loss.cpu().detach().numpy()])
                csvfile.flush()

if __name__ == '__main__':
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    Counting_gs = Counting_2dgs('GS2D/config_ucf_gsplat.yml')
    # Counting_gs = Counting_2dgs('GS2D/config_jhu_gsplat.yml')
    # Counting_gs = Counting_2dgs('GS2D/config_nwpu_gsplat.yml')
    # Counting_gs = Counting_2dgs('GS2D/config_shanghaia_gsplat.yml')
    Counting_gs.batch_saving()
