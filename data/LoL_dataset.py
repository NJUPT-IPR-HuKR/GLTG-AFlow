import os
import subprocess
import torch.utils.data as data
import numpy as np
import time
import torch
import pickle
import cv2
from torchvision.transforms import ToTensor
import random
import torchvision.transforms as T


# import pdb

class LoL_Dataset(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.concat_histeq = all_opt["concat_histeq"] if "concat_histeq" in all_opt.keys() else False
        self.histeq_as_input = all_opt["histeq_as_input"] if "histeq_as_input" in all_opt.keys() else False
        self.log_low = opt["log_low"] if "log_low" in opt.keys() else False
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.use_noise = opt[
            'noise_prob'] if "noise_prob" in opt.keys() else False  # (opt['noise_prob'] and train) if "noise_prob" in opt.keys() else False
        self.noise_prob = opt['noise_prob'] if self.use_noise else None
        self.noise_level = opt['noise_level'] if "noise_level" in opt.keys() else 0
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        self.crop_size = opt.get("GT_size", None)
        # pdb.set_trace()
        if train:
            self.root = os.path.join(self.root, 'our485')
        else:
            self.root = os.path.join(self.root, 'eval15')
        self.pairs = self.load_pairs(self.root)
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path):
        low_list = os.listdir(os.path.join(folder_path, 'low'))
        low_list = filter(lambda x: 'png' in x, low_list)
        pairs = []
        for idx, f_name in enumerate(low_list):
            pairs.append(
                [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name)), cv2.COLOR_BGR2RGB),  # [:, 4:-4, :],
                 cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name)), cv2.COLOR_BGR2RGB),
                 # [:, 4:-4, :],
                 f_name.split('.')[0]])
            # if idx > 10: break
            pairs[-1].append(self.hiseq_color_cv2_img(pairs[-1][0]))
        return pairs

    def hiseq_color_cv2_img(self, img):
        (b, g, r) = cv2.split(img)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        result = cv2.merge((bH, gH, rH))
        return result

    def __getitem__(self, item):
        lr, hr, f_name, his = self.pairs[item]
        if self.histeq_as_input:
            lr = his

        if self.use_crop:
            hr, lr, his = random_crop(hr, lr, his, self.crop_size)

        if self.center_crop_hr_size:
            hr, lr, his = center_crop(hr, self.center_crop_hr_size), center_crop(lr,
                                                                                 self.center_crop_hr_size), center_crop(
                his, self.center_crop_hr_size)

        if self.use_flip:
            hr, lr, his = random_flip(hr, lr, his)

        if self.use_rot:
            hr, lr, his = random_rotation(hr, lr, his)

        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)

        if self.use_noise and random.random() < self.noise_prob:
            lr = torch.randn(lr.shape) * (self.noise_level / 255) + lr
        if self.log_low:
            lr = torch.log(torch.clamp(lr + 1e-3, min=1e-3))
        if self.concat_histeq:
            his = self.to_tensor(his)
            lr = torch.cat([lr, his], dim=0)

        return {'LQ': lr, 'GT': hr, 'LQ_path': f_name, 'GT_path': f_name}


class load_dataset(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.concat_histeq = all_opt["concat_histeq"] if "concat_histeq" in all_opt.keys() else False
        self.histeq_as_input = all_opt["histeq_as_input"] if "histeq_as_input" in all_opt.keys() else False
        self.log_low = opt["log_low"] if "log_low" in opt.keys() else False
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.use_noise = opt[
            'noise_prob'] if "noise_prob" in opt.keys() else False
        self.noise_prob = opt['noise_prob'] if self.use_noise else None
        self.noise_level = opt['noise_level'] if "noise_level" in opt.keys() else 0
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        self.crop_size = opt.get("GT_size", None)
        # pdb.set_trace()
        self.pairs = []
        self.train = train
        for sub_data in ['Real_captured']:  # ['Real_captured']: # :
            if train:
                root = os.path.join(self.root, sub_data, 'Train')
            else:
                root = os.path.join(self.root, sub_data, 'Test')
            self.pairs.extend(self.load_pairs(root))
        self.to_tensor = ToTensor()
        self.gamma_aug = opt['gamma_aug'] if 'gamma_aug' in opt.keys() else False

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path):
        low_list = os.listdir(os.path.join(folder_path, 'Low' if self.train else 'low'))
        low_list = sorted(list(filter(lambda x: 'png' in x, low_list)))
        high_list = os.listdir(os.path.join(folder_path, 'Normal' if self.train else 'high'))
        high_list = sorted(list(filter(lambda x: 'png' in x, high_list)))
        mask_list = os.listdir(r'.\zeromaps')
        mask_list = sorted(list(filter(lambda x: 'png' in x, mask_list)))
        pairs = []
        for idx in range(len(low_list)):
            f_name_low = low_list[idx]
            f_name_high = high_list[idx]
            if self.train:
                mask_index = random.randint(0, len(mask_list) - 1)
            else:
                if idx < len(mask_list):
                    mask_index = idx
                else:
                    mask_index = idx % len(mask_list)

            f_name_mask = mask_list[mask_index]
            pairs.append(
                [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'Low' if self.train else 'low', f_name_low)),
                                cv2.COLOR_BGR2RGB),  # [:, 4:-4, :],
                    cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'Normal' if self.train else 'high', f_name_high)),
                                cv2.COLOR_BGR2RGB),  # [:, 4:-4, :],
                 cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'MASK' if self.train else 'mask', f_name_mask)),
                              cv2.COLOR_BGR2RGB),
                    f_name_high.split('.')[0]])
            # if idx > 10: break
            pairs[-1].append(self.hiseq_color_cv2_img(pairs[-1][0]))
        return pairs

    def hiseq_color_cv2_img(self, img):
        (b, g, r) = cv2.split(img)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        result = cv2.merge((bH, gH, rH))
        return result

    def __getitem__(self, item):
        lr, hr, mask, f_name, his = self.pairs[item]
        mask1 = np.where(mask >= 255, 1, 0)
        lr = lr * (1 - mask1)
        if self.histeq_as_input:
            lr = his

        if self.use_crop:
            hr, lr, mask, his = random_crop(hr, lr, mask, his, self.crop_size)

        if self.center_crop_hr_size:
            hr, lr, his = center_crop(hr, self.center_crop_hr_size), center_crop(lr,
                                                                                 self.center_crop_hr_size), center_crop(
                his, self.center_crop_hr_size)

        if self.use_flip:
            hr, lr, mask, his = random_flip(hr, lr, mask, his)

        if self.use_rot:
            hr, lr, his = random_rotation(hr, lr, his)

        if self.gamma_aug:
            gamma = random.uniform(0.4, 2.8)
            lr = gamma_aug(lr, gamma=gamma)
        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)
        mask = self.to_tensor(mask)
        if self.use_noise and random.random() < self.noise_prob:
            lr = torch.randn(lr.shape) * (self.noise_level / 255) + lr
        if self.log_low:
            lr = torch.log(torch.clamp(lr + 1e-3, min=1e-3))
        if self.concat_histeq:
            his = self.to_tensor(his)
            lr = torch.cat([lr, his], dim=0)

        return {'LQ': lr, 'GT': hr, 'MASK': mask, 'LQ_path': f_name, 'GT_path': f_name, 'MASK_path': f_name}


def random_flip(img, seg,mask, his_eq):
    random_choice = np.random.choice([True, False])
    img = img if random_choice else np.flip(img, 1).copy()
    seg = seg if random_choice else np.flip(seg, 1).copy()
    mask = mask if random_choice else np.flip(mask, 1).copy()
    if his_eq is not None:
        his_eq = his_eq if random_choice else np.flip(his_eq, 1).copy()
    return img, seg,mask, his_eq


def gamma_aug(img, gamma=0):
    max_val = img.max()
    img_after_norm = img / max_val
    img_after_norm = np.power(img_after_norm, gamma)
    return img_after_norm * max_val


def random_rotation(img, seg, his):
    random_choice = np.random.choice([0, 1, 3])
    img = np.rot90(img, random_choice, axes=(0, 1)).copy()
    seg = np.rot90(seg, random_choice, axes=(0, 1)).copy()
    if his is not None:
        his = np.rot90(his, random_choice, axes=(0, 1)).copy()
    return img, seg, his


def random_crop(hr, lr, mask,his_eq, size_hr):
    size_lr = size_hr
    size_mask = size_hr

    size_lr_x = lr.shape[0]
    size_lr_y = lr.shape[1]

    start_x_lr = np.random.randint(low=0, high=(size_lr_x - size_lr) + 1) if size_lr_x > size_lr else 0
    start_y_lr = np.random.randint(low=0, high=(size_lr_y - size_lr) + 1) if size_lr_y > size_lr else 0

    # LR Patch
    lr_patch = lr[start_x_lr:start_x_lr + size_lr, start_y_lr:start_y_lr + size_lr, :]

    # HR Patch
    start_x_hr = start_x_lr
    start_y_hr = start_y_lr
    hr_patch = hr[start_x_hr:start_x_hr + size_hr, start_y_hr:start_y_hr + size_hr, :]

    start_x_mask = start_x_lr
    start_y_mask = start_y_lr
    mask_patch = mask[start_x_mask:start_x_mask + size_mask, start_y_mask:start_y_mask + size_mask, :]
    # HisEq Patch
    his_eq_patch = None
    if his_eq is not None:
        his_eq_patch = his_eq[start_x_hr:start_x_hr + size_hr, start_y_hr:start_y_hr + size_hr, :]
    return hr_patch, lr_patch, mask_patch,  his_eq_patch


def center_crop(img, size):
    if img is None:
        return None
    assert img.shape[1] == img.shape[2], img.shape
    border_double = img.shape[1] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[border:-border, border:-border, :]


def center_crop_tensor(img, size):
    assert img.shape[2] == img.shape[3], img.shape
    border_double = img.shape[2] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[:, :, border:-border, border:-border]
