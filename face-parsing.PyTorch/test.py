#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import glob
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

def vis_parsing_maps(im, parsing_anno, stride, save_im=True, save_path='/content/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    '''
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    '''
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    face_mask = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))
    mouth_mask = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))
    hair_mask =np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))
    #index = np.where(vis_parsing_anno == 1)
    for pi in [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13]:
        face_mask[np.where(vis_parsing_anno == pi)] = [255,255,255]

    #index = np.where(vis_parsing_anno == 11)
    mouth_mask[np.where(vis_parsing_anno == 11)] = [255,255,255]

    #index = np.where(vis_parsing_anno == 17)
    hair_mask[np.where(vis_parsing_anno == 17)] = [255,255,255]

    # Save result or not
    if save_im:
        cv2.imwrite('/content/workspace/data_dst/mask/mask_face.png', face_mask)
        cv2.imwrite('/content/workspace/data_dst/mask/mask_mouth.png', mouth_mask)
        cv2.imwrite('/content/workspace/data_dst/mask/mask_hair.png', hair_mask)

    # return vis_im

def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = './79999_iter.pth'
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in glob.glob(dspth+"/*.*"):
            img = Image.open(image_path)
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))







if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--swap_type", type=str, default="ftm")
    parser.add_argument("--img_root", type=str, default="/data/yuhao.zhu/CelebA-HQ")
    parser.add_argument("--mask_root", type=str, default="/data/yuhao.zhu/CelebAMaskHQ-mask")
    parser.add_argument("--srcID", type=int, default=2332)
    parser.add_argument("--tgtID", type=int, default=2107)
    parser.add_argument("--src_image_path", type=str)
    parser.add_argument("--dst_image_path", type=str)
    parser.add_argument("--dst_mask_path", type=str)
    '''
    
    evaluate(dspth='/content/workspace/data_dst', cp='79999_iter.pth')


