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
                 normalize_std=[0.229, 0.224, 0.225], shot=1, prob=0.7):
        # -------------------load data list,[class,video_name]-------------------
        self.data_dir = data_dir
        self.shot = shot
        self.binary_pair_list = self.get_binary_pair_list()

        new_class_list = []

        fold_list = [0, 1, 2, 3]
        fold_list.remove(fold)
        for fold in fold_list:

            f = open(os.path.join(self.data_dir, 'Binary_map', 'split%1d.txt' % fold))
            while True:
                item = f.readline()
                if item == '':
                    break
                item2 = item.split("  ")
                img_name = item2[0]
                cat = int(item2[1])
                support_img_list = self.binary_pair_list[cat]
                support_names = []
                while True:  # random sample a support data
                    support_name = support_img_list[random.randint(0, len(support_img_list) - 1)]
                    # 在所有 support 列表中随机抽取 support, 直至与 query 匹配为止
                    if img_name != support_name and support_name not in support_names: support_names.append(
                        support_name)
                    if len(support_names) == self.shot: break
                new_class_list.append([img_name, cat, support_names])

        self.new_exist_class_list = new_class_list
        self.initiaize_transformation(normalize_mean, normalize_std, input_size)

        self.input_size = input_size
        self.split = fold
        # self.history_mask_list = [None] * self.__len__()

        self.prob = prob  # probability of sampling history masks=0
        # if self.split == 3:
        #     self.sub_list = list(range(1, 13))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        # elif self.split == 2:
        #     self.sub_list = list(range(1, 9)) + list(range(13, 17))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
        # elif self.split == 1:
        #     self.sub_list = list(range(1, 5)) + list(range(9, 17))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
        # elif self.split == 0:
        #     self.sub_list = list(range(5, 17))  # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]



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
        support_image_list = []
        support_label_list = []
        support_image_th_list = []

        # give a query index,sample a target class first
        query_name = self.new_exist_class_list[index][0]
        sample_class = self.new_exist_class_list[index][1]  # random sample a class in this img
        support_name1 = self.new_exist_class_list[index][2]
        # subcls_list = []
        # # print (self.new_exist_class_list)
        # subcls_list.append(self.sub_list.index(int(sample_class)))
        # print (self.new_exist_class_list)

        # support_img_list = self.binary_pair_list[sample_class]  # all img that contain the sample_class
        # while True:  # random sample a support data
        #     support_name = support_img_list[random.randint(0, len(support_img_list) - 1)]
        #     if support_name != query_name:
        #         break

        # print (query_name,support_name)
        for k in range(self.shot):
            support_name = support_name1[k]
            support_name_rgb = support_name
            support_name_rgb = support_name_rgb.replace('.png', '_rgb.png')
            support_name_th = support_name.replace('.png', '_th.png')
            input_size = self.input_size[0]
            # random scale and crop for support
            scaled_size = int(random.uniform(1, 1.5) * input_size)
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


            ####
            image_th = cv2.imread(os.path.join(self.data_dir, 'seperated_images', support_name_th))
            image_th = Image.fromarray(cv2.cvtColor(image_th, cv2.COLOR_BGR2RGB))

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
                                               support_name)))))

            margin_h = random.randint(0, scaled_size - input_size)
            margin_w = random.randint(0, scaled_size - input_size)
            support_rgb = support_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
            support_mask = support_mask[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
            support_th = support_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
            support_image_list.append(support_rgb)
            support_label_list.append(support_mask)
            support_image_th_list.append(support_th)


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

        # if self.history_mask_list[index] is None:
        #
        #     history_mask = torch.zeros(384, 384).fill_(0.0)
        #
        # else:
        #     if random.random() > self.prob:
        #         history_mask = self.history_mask_list[index]
        #     else:
        #         history_mask = torch.zeros(384, 384).fill_(0.0)

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

        s_xs = support_image_list
        s_ys = support_label_list
        s_ths = support_image_th_list
        s_x = s_xs[0].unsqueeze(0)  # torch.Size([1, 3, 473, 473])
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        s_y = s_ys[0]
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i], s_y], 0)
        s_th = s_ths[0].unsqueeze(0)  # torch.Size([1, 3, 473, 473])
        for i in range(1, self.shot):
            s_th = torch.cat([s_ths[i].unsqueeze(0), s_th], 0)



        return query_rgb ,query_th, query_mask.long(), s_x, s_th, s_y.long(),sample_class-1,index #注意这里的-1，和train.py生成的类别列表保持一致

    def flip(self, flag, img):
        if flag > 0.5:
            return F.hflip(img)
        else:
            return img



    def __len__(self):
        return len(self.new_exist_class_list)
        # return 10


if __name__ == '__main__':
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    dataloader_trn = Dataset(data_dir='../dataset', fold=0,
                             input_size=(384, 384), normalize_mean=mean,
                             shot=5, normalize_std=std)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(dataloader_trn, batch_size=8, shuffle=(train_sampler is None),
                                               num_workers=0, pin_memory=False, sampler=train_sampler,
                                               drop_last=True)

    for idx, (input, input_th, target, s_input, s_input_th, s_mask, subcls, history, index) in enumerate(train_loader):
        print(input.size())
        print(s_input.size())
        print(target.size())
        print(s_mask.size())
        print(subcls.size())


