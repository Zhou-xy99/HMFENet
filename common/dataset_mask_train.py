import cv2
import random
import os
import torchvision
import torch
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F_tensor
import numpy as np
from torch.utils.data import DataLoader
import time



class Dataset(object):


    def __init__(self, data_dir, fold, input_size=[384, 384] , normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225],prob=0.7):
        # -------------------load data list,[class,video_name]-------------------
        self.data_dir = data_dir
        self.new_exist_class_list = self.get_new_exist_class_dict(fold=fold)
        self.initiaize_transformation(normalize_mean, normalize_std, input_size)
        self.binary_pair_list = self.get_binary_pair_list()
        self.input_size = input_size
        self.split = fold
        self.history_mask_list = [None] * self.__len__()
        self.prob = prob  # probability of sampling history masks=0
        # if self.split == 3:
        #     self.sub_list = list(range(1, 13))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        # elif self.split == 2:
        #     self.sub_list = list(range(1, 9)) + list(range(13, 17))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
        # elif self.split == 1:
        #     self.sub_list = list(range(1, 5)) + list(range(9, 17))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
        # elif self.split == 0:
        #     self.sub_list = list(range(5, 17))  # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]



    def get_new_exist_class_dict(self, fold):
        new_exist_class_list = []

        fold_list=[0,1,2,3]
        fold_list.remove(fold)
        for fold in fold_list:

            f = open(os.path.join(self.data_dir, 'Binary_map','split%1d.txt'%fold))
            while True:
                item = f.readline()
                if item == '':
                    break
                item2 = item.split("  ")
                img_name = item2[0]
                cat = int(item2[1])
                new_exist_class_list.append([img_name, cat])
        return new_exist_class_list

    def initiaize_transformation(self, normalize_mean, normalize_std, input_size):
        self.ToTensor = torchvision.transforms.ToTensor()
        self.resize = torchvision.transforms.Resize(input_size)
        self.normalize = torchvision.transforms.Normalize(normalize_mean, normalize_std)

    def rgbnor(self, img):

        means = [0, 0, 0]
        stdevs = [0, 0, 0]
        for i in range(3):
            means[i] = img[:, :, i].mean()
            stdevs[i] = img[:, :, i].std()
        F.normalize(img, means, stdevs, inplace=False)
        return img

    def get_binary_pair_list(self):  # a list store all img name that contain that class
        binary_pair_list = {}
        for Class in range(1, 17):
            binary_pair_list[Class] = self.read_txt(
                os.path.join(self.data_dir, 'Binary_map', '%d.txt' % Class))
        return binary_pair_list

    def read_txt(self, dir):
        f = open(dir)
        out_list = []
        line = f.readline()
        while line:
            out_list.append(line.split()[0])
            line = f.readline()
        return out_list

    def __getitem__(self, index):

        # give an query index,sample a target class first
        query_name = self.new_exist_class_list[index][0]
        sample_class = self.new_exist_class_list[index][1]  # random sample a class in this img
        # subcls_list = []
        # # print (self.new_exist_class_list)
        # subcls_list.append(self.sub_list.index(int(sample_class)))
        # print (self.new_exist_class_list)

        support_img_list = self.binary_pair_list[sample_class]  # all img that contain the sample_class
        while True:  # random sample a support data
            support_name = support_img_list[random.randint(0, len(support_img_list) - 1)]
            if support_name != query_name:
                break

        # print (query_name,support_name)
        support_name_rgb= support_name
        support_name_rgb = support_name_rgb.replace('.png','_rgb.png')########################3
        support_name_th = support_name.replace('.png','_th.png')
        input_size = self.input_size[0]
        # random scale and crop for support
        scaled_size = int(random.uniform(1,1.5)*input_size)
        # scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        # scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        # scale_transform_th = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size],
                                                             interpolation=InterpolationMode.NEAREST)
        scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size],
                                                            interpolation=InterpolationMode.BILINEAR)
        scale_transform_th = torchvision.transforms.Resize([scaled_size, scaled_size],
                                                           interpolation=InterpolationMode.BILINEAR)
        flip_flag = random.random()

        # if self.history_mask_list[index] is None:
        #     # 200,200
        #     history_mask=torch.zeros(200,200).fill_(0.0)   # 创建一个三维数组,用 0.0 填充
        #
        # else:
        #
        #     history_mask=self.history_mask_list[index]

        if self.history_mask_list[index] is None:

            history_mask = torch.zeros(384,384).fill_(0.0)

        else:
            if random.random() > self.prob:
                history_mask = self.history_mask_list[index]
            else:
                history_mask = torch.zeros(384,384).fill_(0.0)
        ####
        image_th = cv2.imread(os.path.join(self.data_dir, 'seperated_images', support_name_th ))
        image_th = Image.fromarray(cv2.cvtColor(image_th,cv2.COLOR_BGR2RGB))

        # support_th = self.normalize(
        #     self.ToTensor(
        #         scale_transform_th(
        #             self.flip(flip_flag,image_th))))

        support_th = self.ToTensor(
            scale_transform_th(
                self.flip(flip_flag, image_th)))
        support_th = self.rgbnor(support_th)

        support_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag,
                              Image.open(
                                  os.path.join(self.data_dir, 'seperated_images', support_name_rgb))))))

        support_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag,
                          Image.open(
                              os.path.join(self.data_dir, 'Binary_map', str(sample_class),
                                           support_name )))))

        margin_h = random.randint(0, scaled_size - input_size)
        margin_w = random.randint(0, scaled_size - input_size)
        support_rgb = support_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_mask = support_mask[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_th = support_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]


        # random scale and crop for query
        scaled_size = input_size  # random.randint(323, 350)
        # scale_transform_th = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        # scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        # scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size],
                                                             interpolation=InterpolationMode.NEAREST)
        scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size],
                                                            interpolation=InterpolationMode.NEAREST)
        scale_transform_th = torchvision.transforms.Resize([scaled_size, scaled_size],
                                                           interpolation=InterpolationMode.BILINEAR)
        flip_flag = 0#random.random()'
        query_name_rgb = query_name
        query_name_rgb = query_name_rgb.replace('.png','_rgb.png')######
        query_name_th = query_name.replace('.png','_th.png')

        image_thq = cv2.imread(os.path.join(self.data_dir, 'seperated_images', query_name_th))
        image_thq = Image.fromarray(cv2.cvtColor(image_thq, cv2.COLOR_BGR2RGB))

        # query_th = self.normalize(
        #     self.ToTensor(
        #         scale_transform_th(
        #             self.flip(flip_flag, image_thq))))

        query_th = self.ToTensor(
            scale_transform_th(
                self.flip(flip_flag, image_thq)))

        query_th = self.rgbnor(query_th)

        query_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag,
                              Image.open(
                                  os.path.join(self.data_dir, 'seperated_images', query_name_rgb ))))))

        query_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag,
                          Image.open(
                              os.path.join(self.data_dir, 'Binary_map', str(sample_class),
                                           query_name)))))

        margin_h = random.randint(0, scaled_size - input_size)
        margin_w = random.randint(0, scaled_size - input_size)

        query_rgb = query_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_mask = query_mask[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_th = query_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

        return query_rgb ,query_th, query_mask.long(), support_rgb,support_th, support_mask.long(),sample_class-1, history_mask,index #注意这里的-1，和train.py生成的类别列表保持一致

    def flip(self, flag, img):
        if flag > 0.5:
            return F.hflip(img)
        else:
            return img



    def __len__(self):
        return len(self.new_exist_class_list)
        # return 10



