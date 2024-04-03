r""" Visualize model predictions """
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

from common import utils


class Visualizer:

    @classmethod
    def initialize(cls, visualize):
        cls.visualize = visualize#True
        if not visualize:
            return

        cls.colors = {'red': (255, 50, 50), 'blue': (102, 140, 255)}
        for key, value in cls.colors.items():
            cls.colors[key] = tuple([c / 255 for c in cls.colors[key]])

        cls.mean_img = [0.485, 0.456, 0.406]
        cls.std_img = [0.229, 0.224, 0.225]
        cls.to_pil = transforms.ToPILImage()
        cls.vis_path = 'visaaaaaa/'
        if not os.path.exists(cls.vis_path): os.makedirs(cls.vis_path)
        global writer

        writer = SummaryWriter("vis")

    @classmethod
    def visualize_prediction_batch(cls, spt_img_b, spt_img_th, spt_mask_b, qry_img_b, qry_th_b, qry_mask_b, pred_mask_b, cls_id_b, batch_idx, iou_b=None):

        spt_img_b = utils.to_cpu(spt_img_b)
        spt_img_th = utils.to_cpu(spt_img_th)
        spt_mask_b = utils.to_cpu(spt_mask_b)
        qry_img_b = utils.to_cpu(qry_img_b)
        qry_th_b = utils.to_cpu(qry_th_b)
        qry_mask_b = utils.to_cpu(qry_mask_b)
        pred_mask_b = utils.to_cpu(pred_mask_b)
        cls_id_b = utils.to_cpu(cls_id_b)

        for sample_idx, (spt_img,spt_th, spt_mask, qry_img, qry_img_th, qry_mask, pred_mask, cls_id) in \
                enumerate(zip(spt_img_b,spt_img_th,spt_mask_b, qry_img_b, qry_th_b, qry_mask_b, pred_mask_b, cls_id_b)):
            iou = iou_b[sample_idx] if iou_b is not None else None
            cls.visualize_prediction(spt_img,spt_th,spt_mask, qry_img, qry_img_th, qry_mask, pred_mask, cls_id, batch_idx,
                                     sample_idx, True, iou)

    @classmethod
    def to_numpy(cls, tensor, type):
        if type == 'img':
            return np.array(cls.to_pil(cls.unnormalize(tensor))).astype(np.uint8)#转为了PIL
        elif type == 'mask':
            return np.array(tensor).astype(np.uint8)
        else:
            raise Exception('Undefined tensor type: %s' % type)

    @classmethod
    def visualize_prediction(cls, spt_imgs,spt_imgs_th,spt_masks, qry_img,qry_img_th,qry_mask, pred_mask, cls_id, batch_idx, sample_idx, label, iou=None):

        spt_color = cls.colors['blue']
        qry_color = cls.colors['red']
        pred_color = cls.colors['red']

        qry_mask = F.interpolate(qry_mask.unsqueeze(0).unsqueeze(0).float(), qry_img.size()[-2:],mode='nearest').squeeze()
        pred_mask = F.interpolate(pred_mask.unsqueeze(0).unsqueeze(0).float(), qry_img.size()[-2:],mode='nearest').squeeze()
        spt_imgs = [cls.to_numpy(spt_img, 'img') for spt_img in spt_imgs]
        spt_imgs1 = [Image.fromarray(spt_img) for spt_img in spt_imgs]#torch.Size([5, 3, 400, 400])-len=5
        spt_imgs_th = [cls.to_numpy(spt_img_th, 'img') for spt_img_th in spt_imgs_th]
        spt_imgs_th1 = [Image.fromarray(spt_img_th) for spt_img_th in spt_imgs_th]
        #(200, 200, 3)
        spt_pils = [cls.to_pil(spt_img) for spt_img in spt_imgs]#len=5
        spt_masks = [cls.to_numpy(spt_mask, 'mask') for spt_mask in spt_masks]#len=5
        spt_masked_pils = [Image.fromarray(cls.apply_mask(spt_img, spt_mask, spt_color)) for spt_img, spt_mask in zip(spt_imgs, spt_masks)]
        qry_img = cls.to_numpy(qry_img, 'img')

        qry_img1 = Image.fromarray(qry_img)#(400, 400, 3)
        qry_pil = cls.to_pil(qry_img)#(400, 400)
        qry_img_th = cls.to_numpy(qry_img_th,"img")
        qry_img_th1 =Image.fromarray(qry_img_th)
        qry_pil_th =cls.to_pil(qry_img_th)
        qry_mask = cls.to_numpy(qry_mask, 'mask')#(400,400)
        pred_mask = cls.to_numpy(pred_mask, 'mask')#(400, 400)
        pred_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), pred_mask.astype(np.uint8), pred_color))#(400, 400)
        pred_masked_pil_th = Image.fromarray(cls.apply_mask(qry_img_th.astype(np.uint8), pred_mask.astype(np.uint8), pred_color))
        qry_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), qry_mask.astype(np.uint8), qry_color))#(400, 400)

        merged_pil = cls.merge_image_pair(spt_imgs1+spt_imgs_th1+spt_masked_pils + [qry_img1,qry_img_th1,pred_masked_pil, pred_masked_pil_th, qry_masked_pil])  #

        iou = iou.item() if iou else 0.0
        img = torch.as_tensor(np.array(merged_pil, copy=True))#torch.Size([473, 1419, 3])
        img = img.permute((2, 0, 1)).unsqueeze(0)
        # img_grid = vutils.make_grid(img, nrow=4, normalize=False, scale_each=False)
        writer.add_images('my_image_batch', img, batch_idx)
        writer.flush()
        merged_pil.save(cls.vis_path + '%d_%d_class-%d_iou-%.2f' % (batch_idx, sample_idx, cls_id, iou) + '.jpg')

    @classmethod
    def merge_image_pair(cls, pil_imgs):
        r""" Horizontally aligns a pair of pytorch tensor images (3, H, W) and returns PIL object """

        canvas_width = sum([pil.size[0] for pil in pil_imgs])
        canvas_height = max([pil.size[1] for pil in pil_imgs])
        canvas = Image.new('RGB', (canvas_width, canvas_height))#带参数的new方法创建image对象

        xpos = 0
        for pil in pil_imgs:
            canvas.paste(pil, (xpos, 0))#填充图像，图片在画布中的左上角位置
            xpos += pil.size[0]

        return canvas

    @classmethod
    def apply_mask(cls, image, mask, color, alpha=0.5):
        r""" Apply mask to the given image. """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    @classmethod
    def unnormalize(cls, img):
        img = img.clone()
        for im_channel, mean, std in zip(img, cls.mean_img, cls.std_img):
            im_channel.mul_(std).add_(mean)
        return img
