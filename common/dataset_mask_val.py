import random
import os
import torchvision
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as F
import torch.nn.functional as F_tensor
import numpy as np
from torch.utils.data import DataLoader
import time

class Dataset(object):


    def __init__(self, data_dir, fold, input_size=[384, 384], normalize_mean=[0, 0, 0],
                 normalize_std=[1, 1, 1],shot=1):

        self.data_dir = data_dir

        self.input_size = input_size

        self.shot=shot
        # self.history_mask_list = [None] * self.__len__()

        #random sample 1000 pairs
        self.chosen_data_list_1 = self.get_new_exist_class_dict(fold=fold)
        # self.chosen_data_list1 = self.chosen_data_list_1[:72]
        chosen_data_list_2 = self.chosen_data_list_1[:]
        chosen_data_list_3 = self.chosen_data_list_1[:]
        # chosen_data_list_4 = self.chosen_data_list_1[:]
        # chosen_data_list_5 = self.chosen_data_list_1[:]
        # chosen_data_list_6 = self.chosen_data_list_1[:]
        random.shuffle(chosen_data_list_2)
        random.shuffle(chosen_data_list_3)
        # random.shuffle(chosen_data_list_4)
        # random.shuffle(chosen_data_list_5)
        # random.shuffle(chosen_data_list_6)

        self.chosen_data_list=self.chosen_data_list_1+chosen_data_list_2+chosen_data_list_3 #+chosen_data_list_4+chosen_data_list_5+chosen_data_list_6
        self.chosen_data_list=self.chosen_data_list[:500]

        self.split = fold
        self.binary_pair_list = self.get_binary_pair_list()#a dict of each class, which contains all imgs that include this class
        self.history_mask_list = [None] * 500
        self.query_class_support_list=[None] * 500


        for index in range (500):#1000
            query_name=self.chosen_data_list[index][0]        # index [0] 为 name
            sample_class=self.chosen_data_list[index][1]      # index [1] 为 class
            support_img_list = self.binary_pair_list[sample_class]  # all img that contain the sample_class
            support_names = []
            while True:  # random sample a support data
                support_name = support_img_list[random.randint(0, len(support_img_list) - 1)]
                # 在所有 support 列表中随机抽取 support, 直至与 query 匹配为止
                if query_name != support_name and support_name not in support_names: support_names.append(support_name)
                if len(support_names) == self.shot: break
            self.query_class_support_list[index]=[query_name,sample_class,support_names]

        if self.split == 3:
            self.sub_val_list = list(range(13, 17))  # [16,17,18,19,20]
        elif self.split == 2:
            self.sub_val_list = list(range(9, 13))  # [6,7,8,9,10]
        elif self.split == 1:
            self.sub_val_list = list(range(5, 9))  # [6,7,8,9,10]
        elif self.split == 0:
            self.sub_val_list = list(range(1, 5))  # [1,2,3,4,5]
        self.initiaize_transformation(normalize_mean, normalize_std, input_size)
        pass


        self.initiaize_transformation(normalize_mean, normalize_std, input_size)
        pass

    def rgbnor(self, img):

        means = [0, 0, 0]
        stdevs = [0, 0, 0]
        for i in range(3):
            means[i] += img[:, :, i].mean()
            stdevs[i] += img[:, :, i].std()
        F.normalize(img, means, stdevs, inplace=False)
        return img

    def get_new_exist_class_dict(self, fold):
        new_exist_class_list = []

        f = open(os.path.join(self.data_dir, 'Binary_map', 'split%1d.txt' % (fold)))
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
        self.normalize = torchvision.transforms.Normalize(normalize_mean, normalize_std)

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

        # give an query index,sample a target class first
        # 一开始采样 500 个得到的一个 list
        query_name = self.query_class_support_list[index][0]
        sample_class = self.query_class_support_list[index][1]  # random sample a class in this img
        support_name1 = self.query_class_support_list[index][2]

        for k in range(self.shot):
            support_name = support_name1[k]

            input_size = self.input_size[0]
            # random scale and crop for support
            scaled_size = int(random.uniform(1, 1.5) * input_size)  # 随机 resize 的大小
            scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size],
                                                                 interpolation=Image.NEAREST)
            scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size],
                                                                interpolation=Image.BILINEAR)
            scale_transform_th = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
            flip_flag = random.random()  # 随机生成的一个实数,它在[0,1)范围内
            support_name_rgb = support_name
            support_name_rgb = support_name_rgb.replace('.png', '_rgb.png')
            support_name_th = support_name.replace('.png', '_th.png')

            image_th = cv2.imread(os.path.join(self.data_dir, 'seperated_images', support_name_th))
            # 读取指定位置文件
            # cv2.imread()用于读取图片文件,加载彩色图片为默认值, 3 channels BGR
            image_th = Image.fromarray(cv2.cvtColor(image_th, cv2.COLOR_BGR2RGB))
            # cv2.cvtColor是颜色空间转换函数, cv2.COLOR_BGR2RGB 将 BGR 格式转换成 RGB 格式
            # Image.fromarray 实现 array 到 image 的转换

            # support_th = self.ToTensor(
            #     scale_transform_th(
            #         self.flip(flip_flag, image_th)))
            #
            # support_th = self.normalize(support_th)

            support_th = self.ToTensor(
                scale_transform_th(
                    self.flip(flip_flag, image_th)))

            support_th = self.rgbnor(support_th)

            support_rgb = self.normalize(self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag,
                              Image.open(
                                  os.path.join(self.data_dir, 'seperated_images', support_name_rgb))))))

            support_mask = self.ToTensor(
                scale_transform_mask(
                    self.flip(flip_flag,
                              Image.open(
                                  os.path.join(self.data_dir, 'Binary_map', str(sample_class), support_name)))))

            margin_h = random.randint(0, scaled_size - input_size)
            margin_w = random.randint(0, scaled_size - input_size)

            support_rgb = support_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
            support_mask = support_mask[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
            support_th = support_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
            support_image_list.append(support_rgb)
            support_label_list.append(support_mask)
            support_image_th_list.append(support_th)

        # random scale and crop for query
        scaled_size = 384  #200

        scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        scale_transform_th = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        flip_flag = 0  # random.random()

        if self.history_mask_list[index] is None:
            # 200,200
            history_mask=torch.zeros(384, 384).fill_(0.0)   # 创建一个三维数组,用 0.0 填充

        else:

            history_mask=self.history_mask_list[index]

        query_name_rgb = query_name
        query_name_rgb = query_name_rgb.replace('.png', '_rgb.png')  ######
        query_name_th = query_name.replace('.png', '_th.png')

        image_thq = cv2.imread(os.path.join(self.data_dir, 'seperated_images', query_name_th))
        # 读取指定位置文件
        # cv2.imread()用于读取图片文件,加载彩色图片为默认值, 3 channels BGR
        image_thq = Image.fromarray(cv2.cvtColor(image_thq, cv2.COLOR_BGR2RGB))
        # cv2.cvtColor是颜色空间转换函数, cv2.COLOR_BGR2RGB 将 BGR 格式转换成 RGB 格式
        # Image.fromarray 实现 array 到 image 的转换

        # query_th = self.ToTensor(
        #     scale_transform_th(
        #         self.flip(flip_flag, image_thq)))
        # # 概率 filp 之后, resize, 之后 h*w*c [0,255] 转换成 c*h*w [0.0,1.0]
        #
        # query_th = self.normalize(query_th)  # 对 query_th RGB 三通道进行 normalization

        query_th = self.ToTensor(
            scale_transform_th(
                self.flip(flip_flag, image_thq)))

        query_th = self.rgbnor(query_th)

        query_rgb = self.normalize(self.ToTensor(
            scale_transform_rgb(
                self.flip(flip_flag,
                          Image.open(
                              os.path.join(self.data_dir, 'seperated_images', query_name_rgb))))))
        # 对 query_rgb 图片进行读取, filp, resize, totensor, normalize, 等预处理

        query_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag,
                          Image.open(
                              os.path.join(self.data_dir, 'Binary_map', str(sample_class),
                                           query_name)))))
        # 对 query_mask 图片除 normalize 外进行预处理

        margin_h = random.randint(0, scaled_size - input_size)  # 0       # 选取随机数，为重新选取 200*200 做准备
        margin_w = random.randint(0, scaled_size - input_size)  # 0

        query_rgb = query_rgb[:, margin_h:margin_h + input_size,
                    margin_w:margin_w + input_size]  # 对于本数据集其实是保留了原始标签，对于其他数据集其实还是裁剪到固定尺寸之后没有random crop
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

        return query_rgb, query_th, query_mask, s_x, s_th, s_y, sample_class-1,history_mask,index #,query_name

    def flip(self, flag, img):
        if flag > 0.5:
            return F.hflip(img)
        else:
            return img  # flag > 0.5 时, 对原图像进行水平翻转

    def __len__(self):
        return 500  # 固定个数为500