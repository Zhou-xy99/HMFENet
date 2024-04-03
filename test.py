import torch.nn as nn
import torch
import argparse
import numpy as np

from torchvision import transforms
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from common import dataset_mask_train, dataset_mask_val
from common.vis import Visualizer

from sklearn.manifold import TSNE

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from collections import OrderedDict
import matplotlib.pyplot as plt
import torch.nn.functional as F


def tSEN(X, Y, scale):

    # 加载数据
    data = X
    label = Y  # 调用函数，获取数据集信息
    print('Starting compute t-SNE Embedding...')
    ts = TSNE(perplexity=30, n_components=2, init='random', random_state=23)  # 11 17
    reslut = ts.fit_transform(data)

    x_min, x_max = np.min(reslut, 0), np.max(reslut, 0)
    reslut = (reslut - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    plt.figure(figsize=(6, 5))  # 创建图形实例
    plt.subplot(111)
    colors = ['c', 'gray', 'pink', 'yellowgreen']

    for i in range(reslut.shape[0]):
        if label[i].item() == 0:
            plt.scatter(reslut[i, 0], reslut[i, 1], c=colors[0], label=str(label[i]))
        elif label[i].item() == 1:
            plt.scatter(reslut[i, 0], reslut[i, 1], c=colors[1], label=str(label[i]))
        elif label[i].item() == 2:
            plt.scatter(reslut[i, 0], reslut[i, 1], c=colors[2], label=str(label[i]))
        elif label[i].item() == 3:
            plt.scatter(reslut[i, 0], reslut[i, 1], c=colors[3], label=str(label[i]))

    plt.xticks()  # 指定坐标的刻度
    plt.yticks()

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), ['Car_Day', 'Car_Night', 'Person_Day', 'Person_Night'], loc='lower right')
    if scale == 8:
        plt.savefig('8_vis.pdf')
    elif scale == 16:
        plt.savefig('16_vis.pdf')
    if scale == 32:
        plt.savefig('32_vis.pdf')
    plt.show()
    ##################################

    # plt.figure(figsize=(6, 5))
    #
    # plt.subplot(111)
    # plt.scatter(reslut[:, 0], reslut[:, 1], c=label, label="t-SNE")
    # plt.legend()
    # plt.savefig('digits_tsne.png')
    # plt.show()

    # 显示图像


def test(epoch, model, dataloader, sub_list, shot):

    utils.fix_randseed(0)
    model.eval()  # 冻结backbone
    average_meter = AverageMeter(sub_list)

    if shot > 1:
        for idx, (input, input_th, target, s_input, s_input_th, s_mask, subcls, history, index) in enumerate(
                dataloader):
            # torch.Size([1, 3, 200, 200]),torch.Size([1, 3, 200, 200]),torch.Size([1, 1, 200, 200]),torch.Size([1, 5, 3, 200, 200]),torch.Size([1, 5, 200, 200])
            logit_mask_agg = 0
            target = target.squeeze(1)
            s_input_th = s_input_th.cuda(non_blocking=True)
            s_input = s_input.cuda(non_blocking=True)
            s_mask = s_mask.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            input_th = input_th.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            subcls = subcls.cuda(non_blocking=True)
            history = history.cuda(non_blocking=True)
            for s_idx in range(shot):
                logit_mask, _, _ = model(input, input_th, s_input[:, s_idx, :, :, :], s_input_th[:, s_idx, :, :, :],
                                         s_mask[:, s_idx, :, :], history)
                # logit_mask = model(input,s_input[:,s_idx, :, :, :], s_mask[:,s_idx, :, :])
                logit_mask_agg += logit_mask.argmax(dim=1).clone()

            # Average & quantize predictions given threshold (=0.5)
            bsz = logit_mask_agg.size(0)  # 1
            max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]  # tensor([5])
            max_vote = torch.stack(
                [max_vote, torch.ones_like(max_vote).long()])  # tensor([[5], [1]]),torch.Size([2, 1])
            max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)  # torch.Size([1, 1, 1])
            pred_mask = logit_mask_agg.float() / max_vote
            pred_mask[pred_mask < 0.5] = 0
            pred_mask[pred_mask >= 0.5] = 1
            for j in range(s_mask.shape[0]):
                sub_index = index[j]
                dataloader_val.history_mask_list[sub_index] = pred_mask[j].data.cpu()

            area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), target)
            average_meter.update(area_inter, area_union, subcls, loss=None)
            average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)


    else:
        for idx, (input, input_th, target, s_input, s_input_th, s_mask, subcls, history, index) in enumerate(
                dataloader):
            # query_np = subcls.cpu().numpy()
            # with open('subcls.txt', 'ab') as f:
            #     np.savetxt(f, query_np)

            s_mask1 = s_mask.squeeze(1)  # torch.Size([1, 200, 200])
            target = target.squeeze(1)
            s_input1 = s_input.squeeze(1)  # torch.Size([1, 3, 200, 200])
            s_input_th1 = s_input_th.squeeze(1)  # torch.Size([1, 3, 200, 200])
            s_input_th1 = s_input_th1.cuda(non_blocking=True)
            s_input1 = s_input1.cuda(non_blocking=True)
            s_mask1 = s_mask1.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            input_th = input_th.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            subcls = subcls.cuda(non_blocking=True)
            history = history.cuda(non_blocking=True)

            # logit_mask_agg, aux_loss1, aux_loss2, query_8, query_16, query_32 = model(input, input_th, s_input1,
            #                                                                           s_input_th1, s_mask1,
            #                                                                           history)  # torch.Size([1, 2, 50, 50]),aux_loss1,aux_loss2
            #
            # query_8 = F.interpolate(query_8, scale_factor=0.0625)
            # query_16 = F.interpolate(query_16, scale_factor=0.125)
            # query_32 = F.interpolate(query_32, scale_factor=0.25)
            #
            # query_8 = query_8.reshape(1, 1, -1).squeeze(1)
            # query_16 = query_16.reshape(1, 1, -1).squeeze(1)
            # query_32 = query_32.reshape(1, 1, -1).squeeze(1)
            # if idx == 420:
            #     features_8 = query_8
            #     features_16 = query_16
            #     features_32 = query_32
            #     labels = subcls.int()
            # elif idx > 420:
            #     features_8 = torch.cat([features_8, query_8], 0)
            #     features_16 = torch.cat([features_16, query_16], 0)
            #     features_32 = torch.cat([features_32, query_32], 0)
            #     labels = torch.cat([labels, subcls.int()], 0)

            # logit_mask_agg = model(input, input_th, s_input1, s_input_th1, s_mask1)
            logit_mask_agg,_,_ = model(input, input_th, s_input1, s_input_th1, s_mask1, target)

            # pred_mask = logit_mask_agg # torch.Size([1, 200, 200])
            pred_mask = logit_mask_agg.argmax(dim=1)  # torch.Size([1, 200, 200])
            for j in range(s_mask.shape[0]):
                sub_index = index[j]
                dataloader_val.history_mask_list[sub_index] = pred_mask[j].data.cpu()
            area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), target)
            average_meter.update(area_inter, area_union, subcls, loss=None)
            average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)
            # Visualize predictions
            Visualizer.visualize_prediction_batch(s_input, s_input_th, s_mask,
                                                      input, input_th, target, pred_mask,
                                                      subcls, idx, iou_b=None)

###########################################################
    # print(features_8.size())
    # print(features_16.size())
    # print(features_32.size())
    # print(labels)
    # tSEN(features_8.cpu(), labels.cpu(), 8)
    # tSEN(features_16.cpu(), labels.cpu(), 16)
    # tSEN(features_32.cpu(), labels.cpu(), 32)
###########################################################
    # Write evaluation results
    average_meter.write_result('Validation', epoch)  # 负责输出一个epoch的结果
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Hypercorrelation Squeeze Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='dataset')
    parser.add_argument('--benchmark', type=str, default='pascal')
    parser.add_argument('--load', type=str, default='logs/_0108_013231.log/best_model.pth')  # 指定training时候文件夹的名字
    parser.add_argument('--bsz', type=int, default=8)  # 20
    parser.add_argument('--lr', type=float, default=0.00005)  # 论文中为1e-3,师兄改成了0.00005
    parser.add_argument('--niter', type=int, default=1)  # train的时候的epoch数2000
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101', 'swin'])
    parser.add_argument('--visualize', action='store_false')
    parser.add_argument('--train_h', type=int, default=473)  # 384
    parser.add_argument('--train_w', type=int, default=473)
    args = parser.parse_args()
    Logger.initialize(args, training=False)  # 开成形成log.txt且已经开始在屏幕上输出内容

    # Model initialization
    # model = HypercorrSqueezeNetwork(args.backbone, False)
    model = torch.load(args.load)
    # model = AttentiveSqueezeNetwork(args.backbone, False)
    Logger.log_params(model)  # 输出模型参数

    if args.fold == 3:
        sub_list = list(range(0, 12))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        sub_val_list = list(range(12, 16))  # [16,17,18,19,20]
    elif args.fold == 2:
        sub_list = list(range(0, 8)) + list(range(12, 16))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
        sub_val_list = list(range(8, 12))  # [6,7,8,9,10]
    elif args.fold == 1:
        sub_list = list(range(0, 4)) + list(range(8, 16))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
        sub_val_list = list(range(4, 8))
    elif args.fold == 0:
        sub_list = list(range(4, 16))  # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        sub_val_list = list(range(0, 4))

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model.to(device)

    # Helper classes (for training) initialization
    Evaluator.initialize()  # cls.ignore_index = 255
    Visualizer.initialize(args.visualize)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Dataset initialization

    dataloader_val = dataset_mask_val.Dataset(data_dir=args.datapath, fold=args.fold,
                                              input_size=(args.train_h, args.train_w), normalize_mean=mean,
                                              normalize_std=std, shot=args.shot)

    val_sampler = None
    val_loader = torch.utils.data.DataLoader(dataloader_val, batch_size=1, shuffle=False,
                                             num_workers=args.nworker, pin_memory=False, sampler=val_sampler)

    # Train HSNet
    best_val_miou = float('-inf')
    best_val_fbiou = float('-inf')
    best_val_loss = float('inf')
    best_epoch = 0
    for epoch in range(args.niter):

        val_loss, val_miou, val_fb_iou = test(epoch, model, val_loader, sub_val_list,
                                              shot=args.shot)

        if val_fb_iou > best_val_fbiou:
            best_val_fbiou = val_fb_iou
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_epoch = epoch
        print("%d epoch ,%.2f is the best miou, %.2f is the best fbiou" % (best_epoch, best_val_miou, best_val_fbiou))

    Logger.tbd_writer.close()
    Logger.info('==================== Finished Test ====================')
