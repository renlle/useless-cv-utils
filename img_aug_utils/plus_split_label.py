# -*- coding: utf-8 -*-
"""

"""
# -*- coding: utf-8 -*-
from DataAugmentation.plus_copy_paste import gaussian_blur_border

"""
数据集0 -> 默认直接复制粘贴 + 高斯模糊噪音
       -> 复制粘贴 50px隔开一份 0.6   + 全图盒子方框滤波
       ->  复制粘贴 50px隔开一份 [高*0.4, 长*1.2] + 高斯模糊噪音
"""
import os, time, sys
import cv2
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, ElementTree
import random

def getColorImg(img, alpha, beta=10):
    """
    alpha 越大越亮, 越小越白
    """
    colored_img = np.uint8(np.clip((alpha * img + beta), 0, 255)) # 限定到最小和最大值范围内
    return colored_img

def get_noisy_img(width, height):
    # 灰色 （47,47,47）- 银色 （192,192,192）  白色255
    noisy_arr = np.random.randn(height, width) * 255  # 生成噪声数据  h * w
    noisy_img_arr = np.stack([noisy_arr, noisy_arr, noisy_arr], axis=2)
    noisy_img = np.uint8(np.clip(noisy_img_arr, 128, 192))
    return noisy_img


def plot(img, xmin, ymin, xmax, ymax):
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot2(img, img_name=""):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # assert False, '在这停顿'



def binding_resize(resize_radio, cut_bbox, x1, y1, x2, y2):
    assert resize_radio > 0 and resize_radio < 2.0
    w = x2 - x1
    h = y2 - y1
    delta_w = int(w * (1 - resize_radio))
    delta_h = int(h * (1 - resize_radio))
    return cv2.resize(cut_bbox, (w - delta_w, h - delta_h)), x1, y1, x2 - delta_w, y2 - delta_h


def binding_resize_two_dems(resize_W_radio, resize_H_radio, cut_bbox, x1, y1, x2, y2):
    assert resize_W_radio > 0 and resize_H_radio > 0
    w = x2 - x1
    h = y2 - y1
    delta_w = int(w * (1 - resize_W_radio))
    delta_h = int(h * (1 - resize_H_radio))
    return cv2.resize(cut_bbox, (w - delta_w, h - delta_h)), x1, y1, x2 - delta_w, y2 - delta_h



def do_one_split(img_path, anno_path, img_outpath, anno_output ,style, dark):
    """
    左右切割, 且加暗色
    styles = ['split_L', 'split_R']
    darkness = [0.3, 0.5]
    """
    img = cv2.imread(img_path)
    img_h, img_w, pixels = img.shape
    cp_img = img.copy()

    # anno
    tree = ET.parse(anno_path)
    root = tree.getroot()
    objects = root.findall("object")
    if len(objects) == 0:
        print('img_path', img_path, '0 anchor, continue')
        return
    x1, y1, x2, y2 = None, None, None, None
    bbox = None
    for obj in objects:
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bbox_w, bbox_h = x2 - x1, y2 - y1

    # L
    if style == 'splitL':
        new_x1, new_y1, new_x2, new_y2 = x1, y1, (x1+x2)//2, y2
        cp_img = cp_img[:, 0:(x1+x2)//2+2]
    else:
        # R
        new_x1, new_y1, new_x2, new_y2 = 0, y1, (x2-x1)//2, y2
        cp_img = cp_img[:, (x1+x2)//2:]


    # gen anno
    bbox_update_new_bbox(new_x1, new_x2, new_y1, new_y2, bbox)

    # 色彩加深
    print(new_x1, new_y1, new_x2, new_y2)
    print(bbox_w, bbox_h)
    cut_img = cp_img[new_y1:new_y2, new_x1:new_x2]
    color_img = getColorImg(cut_img.copy(), alpha=dark)
    gaussian_img = gaussian_blur_border(color_img, img_w, img_h)
    cp_img[new_y1:new_y2, new_x1:new_x2] = gaussian_img


    # plot2(cp_img, img_path)
    # save
    cv2.imwrite(img_outpath, cp_img)
    tree.write(anno_output)


def bbox_update_new_bbox(new_x1, new_x2, new_y1, new_y2, bbox):
    bbox.find('xmin').text=str(int(new_x1))
    bbox.find('ymin').text=str(int(new_y1))
    bbox.find('xmax').text=str(int(new_x2))
    bbox.find('ymax').text=str(int(new_y2))


##################
styles = ['splitL', 'splitR']

def split_label(img_dir, anno_dir, img_write_dir, anno_write_dir):
    if not os.path.exists(img_write_dir):
        os.makedirs(img_write_dir)
        print(' os.makedirs', img_write_dir)
    if not os.path.exists(anno_write_dir):
        os.makedirs(anno_write_dir)
    # img_names = os.listdir(img_dir)


    darkness = [1, 0.4, 0.6, 0.8]

    for img_name in C_class:
        for style in styles:
            for dark in darkness:
                img_path = os.path.join(img_dir, img_name)
                anno_path = os.path.join(anno_dir, img_name.replace(".jpg", '.xml'))
                remark = "splitlabel-{}-dark{}-".format(style, int(dark*10))
                base_name = os.path.splitext(img_name)[0]
                img_write_path = os.path.join(img_write_dir,
                                              f'{remark}{base_name}.jpg')
                anno_write_path = os.path.join(anno_write_dir,
                                               f'{remark}{base_name}.xml')
                print(img_write_path)
                print(anno_write_path)

                do_one_split(img_path, anno_path, img_write_path, anno_write_path, style, dark)



# 轻度剐蹭(片状/不规则漩涡状)
A_class = ['img_' + x + '.jpg' for x in ['1174', '1175']]
# 局部凹陷 (重度)
B_class = ['img_' + x + '.jpg' for x in ['1171', '1172', '1173', '1176', '1177', '1178', '1180']]
# 孔状
C_class = ['img_' + x + '.jpg' for x in ['1179']]

if __name__ == '__main__':
    # 小测试
    data_root = 'E:/cv-datasets/aiwin-gongjiang2021/T1-subway/data/train/WIN_aug_1024/'  # windows 顺带paint一下
    # images_aug_dir = data_root + 'images_aug_1024/img_1172h.jpg'
    # anno_aug_dir = data_root + 'anno_aug_1024/img_1172h.xml'
    images_in_dir = data_root + 'images12/'
    anno_in_dir = data_root + 'annotations_ori12/'
    img_write_dir = data_root + 'plus_split_label/img/'
    anno_write_dir = data_root + 'plus_split_label/anno/'

    split_label(images_in_dir, anno_in_dir, img_write_dir, anno_write_dir)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # real task
    # for alpha in alphas:
    #   copy_paste_main(alpha, images_aug_dir, anno_aug_dir, img_write_dir, anno_write_dir)
