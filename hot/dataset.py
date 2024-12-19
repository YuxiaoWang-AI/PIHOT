import os
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, odgt, opt, **kwargs):
        # parse options
        self.imgSizes = opt.imgSizes
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant

        # parse the input list
        self.parse_input_list(odgt, **kwargs)

        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        segm = torch.from_numpy(np.array(segm)).long()
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p


class TrainDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, batch_per_gpu=1, **kwargs):
        super(TrainDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu
        self.num_class = opt.num_class

        # classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0
        self.if_shuffled = False
        self.a_transform = A.ReplayCompose([
                
                A.RandomRotate90(),
                A.Flip(),
                A.Transpose(),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=45,border_mode=cv2.BORDER_CONSTANT,value=(255,255,255), p=.75),
                
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.2),
                    A.HueSaturationValue(hue_shift_limit=10,sat_shift_limit=10,val_shift_limit=20,p=0.2),
                    A.RGBShift(r_shift_limit=10, g_shift_limit=10,b_shift_limit=10,p=0.2),
                    A.RandomGamma(gamma_limit=(70,150),p=0.2),
                    ],p=0.25),
                
                A.OneOf([
                    A.GaussianBlur(sigma_limit=(0.6, 1.4),p=0.2),
                    A.MotionBlur(blur_limit=5,p=0.2),
                    A.MedianBlur(blur_limit=5,p=0.2),         
                    ], p=0.25),
                
                A.OneOf([
                    A.ISONoise(p=0.3),
                    A.GaussNoise(p=0.3),
                    A.MultiplicativeNoise(p=0.3)
                    ], p=0.25),
                
                A.OneOf([
                    A.Sharpen(alpha=(0.2, 0.5),lightness=(0.9, 1.1)),
                    A.Emboss(),
                    ], p=0.15),
                
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(num_steps = 5, distort_limit = 0.1,p=0.3),
                    A.ElasticTransform(alpha=1, sigma=5,alpha_affine=5,p=0.3)
                    ], p=0.15),
                
                A.OneOf([
                    A.Downscale(scale_min=0.5,scale_max=0.9, p=0.5),
                    A.ImageCompression(quality_lower=30,quality_upper=99, p=0.5)
                    ], p=0.15),
                
                A.Resize(576, 768),
                # A.Resize(256, 256),
                A.RandomCrop(480, 640),
                
            ])
        self.b_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        self.c_transform = A.Compose([
            ToTensorV2(),
        ])

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample) # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample) # h <= w, go to 2nd class

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records

    def random_crop(self, im_h, im_w, crop_h, crop_w):
        res_h = im_h - crop_h
        res_w = im_w - crop_w
        i = random.randint(0, res_h)
        j = random.randint(0, res_w)
        return i, j, crop_h, crop_w

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.seed(index)
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch() 
        batch_height, batch_width = 480, 640
        batch_images = torch.zeros(
            self.batch_per_gpu, 3, batch_height, batch_width)
        batch_segms = torch.zeros(
            self.batch_per_gpu,
            batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate).long()
        batch_part = torch.zeros(
            self.batch_per_gpu,
            batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate).long()
        batch_segm_onehot = torch.zeros(
            self.batch_per_gpu,
            self.num_class).long()
        
        batch_depth= torch.zeros(
            self.batch_per_gpu,
            batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate)
        
        batch_inpaint = torch.zeros(
            self.batch_per_gpu, 3, batch_height, batch_width)
        

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]
            
            # load image and label
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
            part_path = segm_path.replace('annotations', 'segments')
            depth_path = segm_path.replace('annotations', 'depth').replace(".png", ".npy")
            inpaint_depth_path = segm_path.replace('annotations', 'depth').replace(".png", "_inpaint.npy")
            inpaint_path = segm_path.replace('annotations', 'inpainting').replace(".png", "_mask001.png")

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            segm = cv2.imread(segm_path)
            segm_unique_label = np.unique(np.array(segm))

            part = cv2.imread(part_path)
            depth = np.load(depth_path)
            inpaint_depth = np.load(inpaint_depth_path)

            abs_d = np.abs(depth - inpaint_depth)
            abs_d = (abs_d - abs_d.min()) / (abs_d.max() - abs_d.min())
            sc = (255 / abs_d.max())
            abs_d = abs_d * sc
            abs_d = np.clip(abs_d, 0, 255)
            abs_d = np.stack([abs_d, abs_d, abs_d], axis=-1)
            abs_d = abs_d.astype(np.uint8)
            inpaint = cv2.imread(inpaint_path)
            inpaint = cv2.cvtColor(inpaint, cv2.COLOR_BGR2RGB)

            img_data = self.a_transform(image=img, mask=segm)
            inpaint_data = A.ReplayCompose.replay(img_data['replay'], image=inpaint, mask=part)
            abs_d_data = A.ReplayCompose.replay(img_data['replay'], image=abs_d)

            img = img_data["image"]
            inpaint = inpaint_data["image"]
            depth = abs_d_data["image"]
            segm = img_data["mask"]
            part = inpaint_data["mask"]

            img = self.b_transform(image=img)["image"]
            inpaint = self.b_transform(image=inpaint)["image"]
            depth = self.c_transform(image=depth)["image"]
            segm = self.c_transform(image=segm)["image"]
            part = self.c_transform(image=part)["image"]
            depth = depth[0, :, :] / 255.0
            segm = segm[0, :, :]
            part = part[0, :, :]

            depth = torch.nn.functional.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(batch_height // 8, batch_width // 8), mode="nearest").squeeze()
            segm = torch.nn.functional.interpolate(segm.unsqueeze(0).unsqueeze(0), size=(batch_height // 8, batch_width // 8), mode="nearest").squeeze()
            part = torch.nn.functional.interpolate(part.unsqueeze(0).unsqueeze(0), size=(batch_height // 8, batch_width // 8), mode="nearest").squeeze()
            
            # segm_onehot transform, to torch long tensor (num_class)
            segm_onehot = np.zeros(self.num_class) 
            for uid in segm_unique_label:
                segm_onehot[uid] = 1
            segm_onehot = self.segm_transform(segm_onehot) 
            # put into batch arrays
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm
            batch_part[i][:part.shape[0], :part.shape[1]] = part
            batch_segm_onehot[i][:self.num_class] = segm_onehot
            batch_inpaint[i][:, :inpaint.shape[1], :inpaint.shape[2]] = inpaint
            batch_depth[i][:depth.shape[0], :depth.shape[1]] = depth
        

        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms
        output['part_label'] = batch_part
        output['depth_label'] = batch_depth
        output['inpaint_label'] = batch_inpaint
        return output

    def __len__(self):
        return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass


class ValDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, **kwargs):
        super(ValDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset
        self.b_transform = A.Compose([
            A.Resize(480, 640),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        self.c_transform = A.Compose([
            A.Resize(480, 640),
            ToTensorV2(),
        ])
        self.d_transform = A.Compose([
            ToTensorV2(),
        ])

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        depth_path = segm_path.replace('annotations', 'depth').replace(".png", ".npy")
        inpaint_depth_path = segm_path.replace('annotations', 'depth').replace(".png", "_inpaint.npy")
        inpaint_path = segm_path.replace('annotations', 'inpainting').replace(".png", "_mask001.png")


        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        segm = cv2.imread(segm_path)
        
        depth = np.load(depth_path)
        inpaint_depth = np.load(inpaint_depth_path)

        abs_d = np.abs(depth - inpaint_depth)
        abs_d = (abs_d - abs_d.min()) / (abs_d.max() - abs_d.min())
        sc = (255 / abs_d.max())
        abs_d = abs_d * sc
        abs_d = np.clip(abs_d, 0, 255)
        abs_d = np.stack([abs_d, abs_d, abs_d], axis=-1)
        abs_d = abs_d.astype(np.uint8)
        inpaint = cv2.imread(inpaint_path)
        inpaint = cv2.cvtColor(inpaint, cv2.COLOR_BGR2RGB)

        img_data = self.b_transform(image=img, mask=segm)
        inpaint_data = self.b_transform(image=inpaint)
        abs_d_data = self.c_transform(image=abs_d)

        img = img_data["image"]
        inpaint = inpaint_data["image"]
        depth = abs_d_data["image"]
        segm = img_data["mask"]
        
        depth = depth[0, :, :] / 255.0
        segm = segm[:, :, 0]

        depth = torch.nn.functional.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(480 // 8, 640 // 8), mode="nearest").squeeze()
        segm = torch.nn.functional.interpolate(segm.unsqueeze(0).unsqueeze(0), size=(480 // 8, 640 // 8), mode="nearest").squeeze()
        

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = img
        output['depth_label'] = depth
        output['inpaint_label'] = inpaint
        output['seg_label'] = segm
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample


class TestDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, **kwargs):
        super(TestDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset
        self.b_transform = A.Compose([
            A.Resize(480, 640),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        self.c_transform = A.Compose([
            A.Resize(480, 640),
            ToTensorV2(),
        ])
        self.d_transform = A.Compose([
            ToTensorV2(),
        ])
    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        depth_path = segm_path.replace('annotations', 'depth').replace(".png", ".npy")
        inpaint_depth_path = segm_path.replace('annotations', 'depth').replace(".png", "_inpaint.npy")
        inpaint_path = segm_path.replace('annotations', 'inpainting').replace(".png", "_mask001.png")


        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        segm = cv2.imread(segm_path)
        
        depth = np.load(depth_path)
        inpaint_depth = np.load(inpaint_depth_path)

        abs_d = np.abs(depth - inpaint_depth)
        abs_d = (abs_d - abs_d.min()) / (abs_d.max() - abs_d.min())
        sc = (255 / abs_d.max())
        abs_d = abs_d * sc
        abs_d = np.clip(abs_d, 0, 255)
        abs_d = np.stack([abs_d, abs_d, abs_d], axis=-1)
        abs_d = abs_d.astype(np.uint8)
        inpaint = cv2.imread(inpaint_path)
        inpaint = cv2.cvtColor(inpaint, cv2.COLOR_BGR2RGB)

        img_data = self.b_transform(image=img, mask=segm)
        inpaint_data = self.b_transform(image=inpaint)
        abs_d_data = self.c_transform(image=abs_d)

        img = img_data["image"]
        inpaint = inpaint_data["image"]
        depth = abs_d_data["image"]
        segm = img_data["mask"]
        
        depth = depth[0, :, :] / 255.0
        segm = segm[:, :, 0]

        depth = torch.nn.functional.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(480 // 8, 640 // 8), mode="nearest").squeeze()
        segm = torch.nn.functional.interpolate(segm.unsqueeze(0).unsqueeze(0), size=(480 // 8, 640 // 8), mode="nearest").squeeze()
       
        output = dict()
        img_ori = Image.open(image_path).convert('RGB')
        img_ori = img_ori.resize((640, 480))
        output['img_ori'] = np.array(img_ori)
        output['img_data'] = img
        output['depth_label'] = depth
        output['inpaint_label'] = inpaint
        output['seg_label'] = segm
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample
