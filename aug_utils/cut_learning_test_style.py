# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
1. 0->图像大小不变, 内容缩放50%,60%, 缺省背景改为随机灰度图片;
"""
import os, time, sys
import cv2
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET


def get_noisy_img(width, height):
    # 灰色 （47,47,47）- 银色 （192,192,192）  三个值的数值要保持一致!
    noisy_arr = np.random.randn(height, width) * 255  # 生成噪声数据  h * w
    noisy_img_arr = np.stack([noisy_arr, noisy_arr, noisy_arr], axis=2)
    noisy_img = np.uint8(np.clip(noisy_img_arr, 128, 192))
    return noisy_img


def plot(img, xmin, ymin, xmax, ymax):
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def trans_one(alpha, img_path, anno_path, img_outpath, anno_output, offset=100):
    img = cv2.imread(img_path)
    # zoom
    h, w, pixels = img.shape
    zoom_img = cv2.resize(img, (int(w * alpha), int(h * alpha)), interpolation=cv2.INTER_CUBIC)  # w*h
    # gene
    noisy_img = get_noisy_img(w, h)  # 339*1104*3
    noisy_img[offset:offset + int(h * alpha), offset:offset + int(w * alpha)] \
        = zoom_img  # h * w, 3

    # anno
    tree = ET.parse(anno_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    objects = root.findall("object")
    for obj in objects:
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)

        bbox.find('xmin').text = str(int(x1 // 2 + offset))
        bbox.find('ymin').text = str(int(y1 // 2 + offset))
        bbox.find('xmax').text = str(int(x2 // 2 + offset))
        bbox.find('ymax').text = str(int(y2 // 2 + offset))
    # save
    cv2.imwrite(img_outpath, noisy_img)
    tree.write(anno_output)
    #     plot(noisy_img, int(x1)// 2 + offset, int(y1)// 2 + offset, int(x2)// 2 + offset, int(y2)// 2 + offset)


style = ['topleft', 'rightend', 'halfcut']
offset = 100
print('alphas', alphas, 'offset', offset)


def zoom_main(img_dir, anno_dir, img_write_dir, anno_write_dir, data_starts_with=None):
    global alphas
    global offset
    if not os.path.exists(img_write_dir):
        os.makedirs(img_write_dir)
        print(' os.makedirs', img_write_dir)
    if not os.path.exists(anno_write_dir):
        os.makedirs(anno_write_dir)
    img_names = os.listdir(img_dir)
    if data_starts_with is not None:
        # img_names = [x for x in img_names if x.startswith('img_')]
        img_names = [x for x in img_names if x.startswith(data_starts_with)]
    # assert len(img_names) == 96+12, '第0阶段只有96+12个'
    for alpha in alphas:
        for img_name in img_names:
            img_path = os.path.join(img_dir, img_name)
            anno_path = os.path.join(anno_dir, img_name.replace(".jpg", '.xml'))
            remark = f"cut-{alpha}-"
            base_name = os.path.splitext(img_name)[0]
            img_write_path = os.path.join(img_write_dir,
                                          f'{remark}{base_name}.jpg')
            anno_write_path = os.path.join(anno_write_dir,
                                           f'{remark}{base_name}.xml')
            # print(img_write_path)
            # print(anno_write_path)
            trans_one(alpha, img_path, anno_path, img_write_path, anno_write_path, offset)


if __name__ == '__main__':
    # 小测试
    data_root = 'E:/cv-datasets/aiwin-gongjiang2021/T1-subway/data/train/WIN_aug_1024/'  # windows 顺带paint一下
    # images_aug_dir = data_root + 'images_aug_1024/img_1172h.jpg'
    # anno_aug_dir = data_root + 'anno_aug_1024/img_1172h.xml'
    images_aug_dir = data_root + 'images12/'
    anno_aug_dir = data_root + 'annotations_ori12/'
    img_write_dir = data_root + 'cut_learning/'
    anno_write_dir = data_root + 'cut_learning_XML/'

    zoom_main(images_aug_dir, anno_aug_dir, img_write_dir, anno_write_dir,data_starts_with='zoom')
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # real task
    # for alpha in alphas:
    #   zoom_main(alpha, images_aug_dir, anno_aug_dir, img_write_dir, anno_write_dir)
