# USELESS-CV-UTILS

## 介绍
整理个人日常的计算机视觉算法相关的有趣的工具类代码等;
- gitee首发: https://gitee.com/renlle/useless-cv-utils
- github每周自动同步一次: 
- 知乎专栏：https://www.zhihu.com/column/c_1438647693480062976


## 分支说明
- master  稳定分支, 经过测试，测试效果见每个package下的in_imgs和out_imgs的对照
- develop  不稳定, 随时更新代码


## 组织模块

### img_aug_utils
模块简介: 特别的数据增强工具类
- cut_learning_test_style.py
- plus_copy_paste.py 简单版本的
- plus_split_label.py   把检测样本进行分割成两部分,再加入训练集

### img_mask2bbox_utils
模块简介: 尝试把`Type-I RSDDs dataset`的纯mask图像转为bbox, 并且添加舍弃低质量的mask的阈值
TODO
...
