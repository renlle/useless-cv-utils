# -*- coding: utf-8 -*-
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


def gaussian_blur_border(img, width, height):
    data = (width, height)
    img_copy = img.copy()
    imgCanny = cv2.Canny(img, *data)
    # 创建矩形结构
    num = 2
    g = cv2.getStructuringElement(cv2.MORPH_RECT, (num, num))
    g2 = cv2.getStructuringElement(cv2.MORPH_RECT, (num * 2, num * 2))
    # 膨化处理
    # 更细腻
    img_dilate = cv2.dilate(imgCanny, g)
    # 更粗大
    img_dilate2 = cv2.dilate(imgCanny, g2)

    shape = img_dilate.shape
    # 提取
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img_dilate2[i, j] == 0:  # 二维定位到三维
                img[i, j] = [0, 0, 0]

    # lens = height//2+1
    lens = 5
    dst = cv2.GaussianBlur(img, (lens, lens), 0, 0, cv2.BORDER_DEFAULT)

    for i in range(shape[0]):
        for j in range(shape[1]):
            if img_dilate[i, j] != 0:  # 二维定位到三维
                img_copy[i, j] = dst[i, j]

    return img_copy


def cp_gaosi_maski_another(img_path, anno_path, img_outpath, anno_output, style):
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
    for obj in objects:
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bbox_w, bbox_h = x2 - x1, y2 - y1
    # do cp
    random.seed(23333)
    # up_float = random.randint(-20, 20)
    up_float = 0
    r_float = random.randint(-10, 20) + 50 + bbox_w
    if abs(x2 - img_w) < r_float + 50: r_float *= -1
    new_x1, new_y1, new_x2, new_y2 = x1 + r_float, y1 + up_float, x2 + r_float, y2 + up_float
    print('>>x1, y1, x2, y2:', x1, y1, x2, y2)
    print('>>new_x1, new_y1, new_x2, new_y2:', new_x1, new_y1, new_x2, new_y2)

    # cut  h * w
    cut_bbox = cp_img[y1:y2, x1:x2]
    # 做一下高斯模糊
    gaussian_cut_bbox = gaussian_blur_border(cut_bbox.copy(), x2 - x1, y2 - y1)

    cp_img[new_y1:new_y2, new_x1:new_x2] = gaussian_cut_bbox

    # gen anno
    tree_add_new_bbox(new_x1, new_x2, new_y1, new_y2, root)

    # plot2(cp_img, img_path)
    # save
    cv2.imwrite(img_outpath, cp_img)
    tree.write(anno_output)


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


def box_filter(img_path, anno_path, img_outpath, anno_output, style):
    """
    盒子方框滤波
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
    for obj in objects:
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bbox_w, bbox_h = x2 - x1, y2 - y1
    # do cp
    random.seed(23333)
    # up_float = random.randint(-20, 20)
    up_float = 0
    r_float = random.randint(-10, 20) + 50 + bbox_w
    if abs(x2 - img_w) < r_float + 50: r_float *= -1
    new_x1, new_y1, new_x2, new_y2 = x1 + r_float, y1 + up_float, x2 + r_float, y2 + up_float
    print('>>x1, y1, x2, y2:', x1, y1, x2, y2)
    print('>>new_x1, new_y1, new_x2, new_y2:', new_x1, new_y1, new_x2, new_y2)

    # cut  h * w
    cut_bbox = cp_img[y1:y2, x1:x2]

    cut_bbox, new_x1, new_y1, new_x2, new_y2 = binding_resize(0.6, cut_bbox.copy(), new_x1, new_y1, new_x2, new_y2)
    print('>>new_x1, new_y1, new_x2, new_y2, cut_bbox.shape:', new_x1, new_y1, new_x2, new_y2, cut_bbox.shape)

    cp_img[new_y1:new_y2, new_x1:new_x2] = cut_bbox

    # 全局方框滤波
    cp_img = cv2.boxFilter(cp_img, -1, (5, 5), normalize=1)

    # gen anno
    tree_add_new_bbox(new_x1, new_x2, new_y1, new_y2, root)

    # plot2(cp_img, img_path)
    # save
    cv2.imwrite(img_outpath, cp_img)
    tree.write(anno_output)


def cp_gaosi_fold(img_path, anno_path, img_outpath, anno_output, style, resize_W_radio, resize_H_radio):
    """
    高斯后折叠
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
    for obj in objects:
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bbox_w, bbox_h = x2 - x1, y2 - y1
    # do cp
    random.seed(23333)
    # up_float = random.randint(-20, 20)
    up_float = 0
    r_float = random.randint(-10, 20) + 50 + bbox_w
    if abs(x2 - img_w) < r_float + 50: r_float *= -1
    new_x1, new_y1, new_x2, new_y2 = x1 + r_float, y1 + up_float, x2 + r_float, y2 + up_float
    print('>>x1, y1, x2, y2:', x1, y1, x2, y2)
    print('>>new_x1, new_y1, new_x2, new_y2:', new_x1, new_y1, new_x2, new_y2)

    # cut  h * w
    cut_bbox = cp_img[y1:y2, x1:x2]
    # 做一下高斯模糊
    gaussian_cut_bbox = gaussian_blur_border(cut_bbox.copy(), x2 - x1, y2 - y1)

    # 折叠, 并更新bbox  [高*0.3, 长*1.2]
    fold_cut_bbox, new_x1, new_y1, new_x2, new_y2 = binding_resize_two_dems(resize_W_radio, resize_H_radio, gaussian_cut_bbox.copy(), new_x1, new_y1, new_x2,
                                                                       new_y2)

    cp_img[new_y1:new_y2, new_x1:new_x2] = fold_cut_bbox

    # gen anno
    tree_add_new_bbox(new_x1, new_x2, new_y1, new_y2, root)

    # plot2(cp_img, img_path)
    # save
    cv2.imwrite(img_outpath, cp_img)
    tree.write(anno_output)


def tree_add_new_bbox(new_x1, new_x2, new_y1, new_y2, root):
    new_obj = Element('object')
    # <name>defect</name>
    e_name = Element('name')
    e_name.text = 'defect'
    new_obj.append(e_name)
    root.append(new_obj)
    e_bndbox = Element('bndbox')
    new_obj.append(e_bndbox)
    e_xmin, e_xmax, e_ymin, e_ymax = Element('xmin'), Element('xmax'), Element('ymin'), Element('ymax')
    e_xmin.text = str(new_x1)
    e_xmax.text = str(new_x2)
    e_ymin.text = str(new_y1)
    e_ymax.text = str(new_y2)
    e_bndbox.append(e_xmin)
    e_bndbox.append(e_xmax)
    e_bndbox.append(e_ymin)
    e_bndbox.append(e_ymax)
    return root


##################

def copy_paste_main(img_dir, anno_dir, img_write_dir, anno_write_dir):
    if not os.path.exists(img_write_dir):
        os.makedirs(img_write_dir)
        print(' os.makedirs', img_write_dir)
    if not os.path.exists(anno_write_dir):
        os.makedirs(anno_write_dir)
    img_names = os.listdir(img_dir)
    for style in ['gaosi_border', 'box_filter', 'gaosi_folder']:
        for img_name in img_names:
            img_path = os.path.join(img_dir, img_name)
            anno_path = os.path.join(anno_dir, img_name.replace(".jpg", '.xml'))
            remark = f"copy_paste-{style}-"
            base_name = os.path.splitext(img_name)[0]
            img_write_path = os.path.join(img_write_dir,
                                          f'{remark}{base_name}.jpg')
            anno_write_path = os.path.join(anno_write_dir,
                                           f'{remark}{base_name}.xml')
            print(img_write_path)
            print(anno_write_path)

            if style == 'gaosi_border':
                cp_gaosi_maski_another(img_path, anno_path, img_write_path, anno_write_path, style)
            elif style == 'box_filter':
                box_filter(img_path, anno_path, img_write_path, anno_write_path, style)
            elif style == 'gaosi_folder':
                cp_gaosi_fold(img_path, anno_path, img_write_path, anno_write_path, style,resize_W_radio=1.2, resize_H_radio=0.4)
            else:
                assert False


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
    img_write_dir = data_root + 'CP/img/'
    anno_write_dir = data_root + 'CP/anno/'

    copy_paste_main(images_in_dir, anno_in_dir, img_write_dir, anno_write_dir)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # real task
    # for alpha in alphas:
    #   copy_paste_main(alpha, images_aug_dir, anno_aug_dir, img_write_dir, anno_write_dir)
