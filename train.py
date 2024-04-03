r""" Hypercorrelation Squeeze training (validation) code """
import argparse

import torch.optim as optim
import torch.nn as nn
import torch

from model.hsnet10 import HypercorrSqueezeNetwork
from model.DCAMA import DCAMA
from model.asnet import AttentiveSqueezeNetwork
# from model.sagnn import SAGNN
from model.asgnet import ASGN

from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from common import dataset_mask_train, dataset_mask_val, dataset_mask_train_pro
from common.vis import Visualizer


def train(epoch, model, dataloader, optimizer, sub_list,training,shot):
    r""" Train Net """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    # model.train() if training else model.module.eval()#冻结backbone
    model.module.train_mode() if training else model.module.eval()#冻结backbone
    average_meter = AverageMeter(sub_list)


    if (shot>1) and (not training):
        for idx, (input, input_th, target, s_input, s_input_th, s_mask, subcls,history,index) in enumerate(dataloader):
            #torch.Size([1, 3, 200, 200]),torch.Size([1, 3, 200, 200]),torch.Size([1, 1, 200, 200]),torch.Size([1, 5, 3, 200, 200]),torch.Size([1, 5, 200, 200])
            # print(input.size())# [1,3,384,384]
            # print(s_input.size())# [1,5,3,384,384]
            # print(subcls.size())# [1]
            logit_mask_agg = 0
            logit=0
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
                logit_mask,_,_= model(input, input_th,s_input[:,s_idx, :, :, :], s_input_th[:,s_idx, :, :, :], s_mask[:,s_idx, :, :],history)
                # logit_mask = model(input, input_th,s_input[:,s_idx, :, :, :], s_input_th[:,s_idx, :, :, :], s_mask[:,s_idx, :, :],history)
                # logit_mask = model(input,s_input[:,s_idx, :, :, :], s_mask[:,s_idx, :, :])
                # logit_mask = model(input, input_th,s_input[:,s_idx, :, :, :], s_input_th[:,s_idx, :, :, :], s_mask[:,s_idx, :, :])

                logit_mask_agg += logit_mask.argmax(dim=1).clone()
            # logit_mask_agg = model(input, input_th, s_input, s_input_th, s_mask)  #用于基于原型的模型
            # Average & quantize predictions given threshold (=0.5)
            bsz = logit_mask_agg.size(0)  # 1
            max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]  # tensor([5])
            max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])  # tensor([[5], [1]]),torch.Size([2, 1])
            max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)  # torch.Size([1, 1, 1])
            pred_mask = logit_mask_agg.float() / max_vote
            pred_mask[pred_mask < 0.5] = 0
            pred_mask[pred_mask >= 0.5] = 1
            for j in range(s_mask.shape[0]):
                sub_index = index[j]
                dataloader_val.history_mask_list[sub_index] =pred_mask[j].data.cpu()

            area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), target)
            average_meter.update(area_inter, area_union, subcls, loss=None)
            average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)


    else:
        for idx, (input, input_th, target, s_input, s_input_th, s_mask, subcls,history,index) in enumerate(dataloader):
            s_mask1 = s_mask.squeeze(1)#torch.Size([1, 200, 200])
            target = target.squeeze(1)
            s_input1 = s_input.squeeze(1)#torch.Size([1, 3, 200, 200])
            s_input_th1 = s_input_th.squeeze(1)#torch.Size([1, 3, 200, 200])
            s_input_th1 = s_input_th1.cuda(non_blocking=True)
            s_input1 = s_input1.cuda(non_blocking=True)
            s_mask1 = s_mask1.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            input_th = input_th.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            subcls = subcls.cuda(non_blocking=True)
            history = history.cuda(non_blocking=True)
            logit_mask_agg,aux_loss1,aux_loss2 = model(input, input_th, s_input1, s_input_th1, s_mask1,history)  # torch.Size([1, 2, 50, 50]),aux_loss1,aux_loss2
            # logit_mask_agg = model(input, input_th, s_input1, s_input_th1, s_mask1,history)
            # logit_mask_agg, main_loss, aux_loss = model(input, input_th, s_input1, s_input_th1, s_mask1, target)
            # logit_mask_agg = model(input, input_th, s_input1, s_input_th1, s_mask1)

            # pred_mask = logit_mask_agg  # torch.Size([1, 200, 200])
            pred_mask = logit_mask_agg.argmax(dim=1)  # torch.Size([1, 200, 200])


            if training:
                for j in range(s_mask.shape[0]):
                    sub_index = index[j]
                    dataloader_trn.history_mask_list[sub_index] = pred_mask[j].data.cpu()
                # loss = main_loss + aux_loss
                loss = model.module.compute_objective(logit_mask_agg, target)
                loss = loss + 0.3*aux_loss1 + 0.3*aux_loss2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                area_inter, area_union = Evaluator.classify_prediction(pred_mask, target)
                average_meter.update(area_inter, area_union, subcls, loss.detach().clone())
                average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)  # 每50个iteration显示一次结果

            else:
                for j in range(s_mask.shape[0]):
                    sub_index = index[j]
                    dataloader_val.history_mask_list[sub_index] = pred_mask[j].data.cpu()
                area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), target)
                average_meter.update(area_inter, area_union, subcls, loss=None)
                average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)



    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)#负责输出一个epoch的结果
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou




if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Hypercorrelation Squeeze Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='dataset')
    parser.add_argument('--benchmark', type=str, default='pascal')
    parser.add_argument('--logpath', type=str, default='')#指定training时候文件夹的名字
    parser.add_argument('--bsz', type=int, default=8)#20
    parser.add_argument('--lr', type=float, default=0.00005)#论文中为1e-3,师兄改成了0.00005
    parser.add_argument('--niter', type=int, default=100) #train的时候的epoch数2000
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='swin', choices=['vgg16', 'resnet50', 'resnet101', 'swin'])
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--train_h', type=int, default=384) #384,473,200
    parser.add_argument('--train_w', type=int, default=384)
    args = parser.parse_args()
    Logger.initialize(args, training=True)#开成形成log.txt且已经开始在屏幕上输出内容

    # Model initialization
    # model = HypercorrSqueezeNetwork(args.backbone, False)
    model = DCAMA(args.backbone, False)
    # model = AttentiveSqueezeNetwork(args.backbone, False)
    # model = SAGNN()
    # model = ASGN(args)
    Logger.log_params(model)#输出模型参数

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
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Helper classes (for training) initialization
    optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])
    Evaluator.initialize()#cls.ignore_index = 255
    Visualizer.initialize(args.visualize)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Dataset initialization

    dataloader_trn = dataset_mask_train.Dataset(data_dir=args.datapath, fold=args.fold,input_size=(args.train_h,args.train_w), normalize_mean=mean,
                            normalize_std=std)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(dataloader_trn, batch_size=args.bsz, shuffle=(train_sampler is None),
                                               num_workers=args.nworker, pin_memory=False, sampler=train_sampler,drop_last=True)

    dataloader_val = dataset_mask_val.Dataset(data_dir=args.datapath, fold=args.fold,input_size=(args.train_h,args.train_w), normalize_mean=mean,
                            normalize_std=std,shot=args.shot)


    val_sampler = None
    val_loader = torch.utils.data.DataLoader(dataloader_val, batch_size=1, shuffle=False,
                                             num_workers=args.nworker, pin_memory=False, sampler=val_sampler)

    # Train HSNet
    best_val_miou = float('-inf')
    best_val_fbiou = float('-inf')
    best_val_loss = float('inf')
    best_epoch = 0
    for epoch in range(args.niter):

        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, train_loader, optimizer,sub_list, training=True,shot=1)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, val_loader, optimizer,sub_val_list, training=False,shot=args.shot)

        if val_fb_iou > best_val_fbiou:
            best_val_fbiou = val_fb_iou
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_epoch = epoch
            Logger.save_model_miou(model, epoch, val_miou)#保存模型
        print("%d epoch ,%.2f is the best miou, %.2f is the best fbiou" % (best_epoch, best_val_miou, best_val_fbiou))


        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
