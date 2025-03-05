
import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import copy
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from tqdm import tqdm
import albumentations as A


def gram_matrix(x):
    b, c, h, w = x.shape
    features = x.view(b, c, h*w)  # shape: (b, c, N)
    gram = torch.bmm(features, features.transpose(1, 2)) / (c * h * w)
    return gram


import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import copy
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from tqdm import tqdm
import albumentations as A



def get_features(x, module_type, stage):
    """
    获取特定层的输出特征。
    
    参数:
        x (torch.Tensor): 输入特征图，形状为 [batch, channels, height, width]。
        module_type (str): 模块类型（如 "Detect", "Pose", "Segment"）。
        stage (int): 当前层数。
    
    返回:
        out_feas (torch.Tensor): 保存的特征图列表（如果层数属于 [2, 4, 6, 8, 9]）。
    """
    # 如果模块类型是 "Detect", "Pose", "Segment"，直接返回
    for m in ["Detect", "Pose", "Segment"]:
        if m in module_type:
            return None

    # 获取输入特征图的形状
    batch, channels, height, width = x.shape  # batch, channels, height, width

    # 初始化特征图列表
    out_feas_list = []
    # print(f"daca.py ⚠️ Computing features at stage {stage}") 
    
    # 如果层数属于 [2, 4, 6, 8, 9]，保存特征图
    if stage in [0]:
        print(f"Saving features at stage {stage}")  # 打印当前层数
        out_feas_list.append(x)

    # 将特征图列表转换为张量
    if out_feas_list:  # 如果列表不为空
        out_feas = torch.stack(out_feas_list)  # 将列表中的张量堆叠为一个张量
    else:
        out_feas = None  # 如果没有保存特征图，返回 None

    return out_feas


def get_best_region(out, imgs_t):
    # imgs_t.shape：torch.Size([4, 3, 640, 640])
    region_t = imgs_t[:, :, 0:int(imgs_t.shape[3]/2), 0:int(imgs_t.shape[2]/2)] # initialize in case no bboxes are detected
    best_side = 'topleft'  # initialize in case no bboxes are detected 左上角
    if out.shape[0] > 0:
        bboxes_target = copy.deepcopy(out) # [batch_id, class_id, x, y, w, h, conf] (16,7)
        # 筛选左上角区域 (topleft)： numpy < float 320.0
        # bboxes_target[:, 2] < imgs_t.shape[2]/2 → 目标中心点 cx 在左半部分
        # bboxes_target[:, 3] < imgs_t.shape[3]/2 → 目标中心点 cy 在上半部分
        bboxes_target_topleft = bboxes_target[bboxes_target[:, 2] < imgs_t.shape[2]/2, :]
        bboxes_target_topleft = bboxes_target_topleft[bboxes_target_topleft[:, 3] < imgs_t.shape[3]/2, :]
        # 筛选左下角 (bottomleft)
        bboxes_target_bottomleft = bboxes_target[bboxes_target[:, 2] < imgs_t.shape[2]/2, :]
        bboxes_target_bottomleft = bboxes_target_bottomleft[bboxes_target_bottomleft[:, 3] > imgs_t.shape[3]/2, :]
        # 筛选右下角 (bottomright)
        bboxes_target_bottomright = bboxes_target[bboxes_target[:, 2] > imgs_t.shape[2]/2, :]
        bboxes_target_bottomright = bboxes_target_bottomright[bboxes_target_bottomright[:, 3] > imgs_t.shape[3]/2, :]
        # 筛选右上角 (topright)
        bboxes_target_topright = bboxes_target[bboxes_target[:, 2] > imgs_t.shape[2]/2, :]
        bboxes_target_topright = bboxes_target_topright[bboxes_target_topright[:, 3] < imgs_t.shape[3]/2, :]
        # 计算每个区域目标框的平均置信度 conf
        conf_topleft = np.mean(bboxes_target_topleft[:, -1]) if len(bboxes_target_topleft)>0 else 0
        conf_bottomleft = np.mean(bboxes_target_bottomleft[:, -1]) if len(bboxes_target_bottomleft)>0 else 0
        conf_bottomright = np.mean(bboxes_target_bottomright[:, -1]) if len(bboxes_target_bottomright)>0 else 0
        conf_topright = np.mean(bboxes_target_topright[:, -1]) if len(bboxes_target_topright)>0 else 0

        if bboxes_target.shape[0]>0:
            # 存储四个方向的名称
            side = ['topleft', 'bottomleft', 'bottomright', 'topright'] 
            # 存储四个方向的目标框数据 
            region_bboxes = [bboxes_target_topleft, bboxes_target_bottomleft, bboxes_target_bottomright, bboxes_target_topright]
            # 存储四个区域的置信度
            conf = [conf_topleft, conf_bottomleft, conf_bottomright, conf_topright]
            # 找到置信度最高的区域索引
            id_best = conf.index(max(conf))
            # 更新最佳区域名称
            best_side = side[id_best]
            # 获取该区域的目标框
            best_bboxes = region_bboxes[id_best]
        else:
            best_bboxes = []

        if best_bboxes.shape[0]>0:
            out = copy.deepcopy(best_bboxes)
            out = torch.from_numpy(out) # 转换为tensor

        # 对每个区域进行坐标裁剪，防止 bbox 超出选定区域
        if best_side == 'topleft' and best_bboxes.shape[0] > 0: 
            # 遍历 out 里的 bbox，如果超出该区域范围，则进行修正（调整 cx, cy, w, h）
            region_t = imgs_t[:, :, 0:int(imgs_t.shape[3]/2), 0:int(imgs_t.shape[2]/2)]
            # clip bboxes that exceed the selected area
            for o in range(len(out)):
                if out[o, 3] + out[o, 5]/2 > imgs_t.shape[3]/2:
                    new_h = imgs_t.shape[3]/2 - out[o, 3] + out[o, 5]/2
                    new_cy =  imgs_t.shape[3]/2 - new_h/2
                    out[o, 3] = int(new_cy)
                    out[o, 5] = int(new_h)
                if out[o, 2] + out[o, 4]/2 > imgs_t.shape[2]/2:
                    new_w = imgs_t.shape[2]/2 - out[o, 2] + out[o, 4]/2
                    new_cx =  imgs_t.shape[2]/2 - new_w/2
                    out[o, 2] = int(new_cx)
                    out[o, 4] = int(new_w)
            
        elif best_side == 'bottomleft' and best_bboxes.shape[0] > 0: 
            region_t = imgs_t[:, :, int(imgs_t.shape[3]/2):, 0:int(imgs_t.shape[2]/2)]
            # clip bboxes that exceed the selected area
            for o in range(len(out)):
                if out[o, 3] - out[o, 5]/2 < imgs_t.shape[3]/2:
                    new_h = out[o, 3] - imgs_t.shape[3]/2 + out[o, 5]/2
                    new_cy =  imgs_t.shape[3]/2 + new_h/2
                    out[o, 3] = int(new_cy)
                    out[o, 5] = int(new_h)
                if out[o, 2] + out[o, 4]/2 > imgs_t.shape[2]/2:
                    new_w = imgs_t.shape[2]/2 - out[o, 2] + out[o, 4]/2
                    new_cx =  imgs_t.shape[2]/2 - new_w/2
                    out[o, 2] = int(new_cx)
                    out[o, 4] = int(new_w)

        elif best_side == 'bottomright' and best_bboxes.shape[0] > 0: 
            region_t = imgs_t[:, :, int(imgs_t.shape[3]/2):, int(imgs_t.shape[2]/2):]                       
            # clip bboxes that exceed the selected area
            for o in range(len(out)):
                if out[o, 3] - out[o, 5]/2 < imgs_t.shape[3]/2:
                    new_h = out[o, 3] - imgs_t.shape[3]/2 + out[o, 5]/2
                    new_cy =  imgs_t.shape[3]/2 + new_h/2
                    out[o, 3] = int(new_cy)
                    out[o, 5] = int(new_h)
                if out[o, 2] - out[o, 4]/2 < imgs_t.shape[2]/2:
                    new_w = out[o, 2] - imgs_t.shape[2]/2 + out[o, 4]/2
                    new_cx =  imgs_t.shape[2]/2 + new_w/2
                    out[o, 2] = int(new_cx)
                    out[o, 4] = int(new_w)

        elif best_side == 'topright' and best_bboxes.shape[0] > 0: 
            region_t = imgs_t[:, :, 0:int(imgs_t.shape[3]/2), int(imgs_t.shape[2]/2):]
            # clip bboxes that exceed the selected area
            for o in range(len(out)):
                if out[o, 3] + out[o, 5]/2 > imgs_t.shape[3]/2:
                    new_h = imgs_t.shape[3]/2 - out[o, 3] + out[o, 5]/2
                    new_cy =  imgs_t.shape[3]/2 - new_h/2
                    out[o, 3] = int(new_cy)
                    out[o, 5] = int(new_h)
                if out[o, 2] - out[o, 4]/2 < imgs_t.shape[2]/2:
                    new_w = out[o, 2] - imgs_t.shape[2]/2 + out[o, 4]/2
                    new_cx =  imgs_t.shape[2]/2 + new_w/2
                    out[o, 2] = int(new_cx)
                    out[o, 4] = int(new_w)

    else:
        out = torch.empty([0,7])  # 如果没有检测到目标，返回空张量
    
    # 最佳区域的图像部分,
    # 最佳区域中的目标框信息（格式 [batch_id, class, cx, cy, w, h, conf]）
    # 最佳区域的位置（topleft，bottomleft，bottomright，topright）
    return region_t, out, best_side 

def transform_img_bboxes(out, best_side, region_t, transform_):
    '''
    out 原始的检测框(bounding boxes) 形状为 (N, 7)，其中 N 是目标的数量。
    best_side:选定的最佳区域（'topleft', 'bottomleft', 'bottomright', 'topright'）。
    region_t:从原始图像裁剪出的最佳区域图像。 region_t.shape = torch.Size([4, 3, 320, 320])
    transform_:一个用于数据增强的变换函数。
    '''
    out_ = copy.deepcopy(out) # 作用是创建 out 的深拷贝，确保不会修改原始数据
    
    # fit the coordinates into the region-level reference instead of whole image
    # 目标框 out_ 是基于整张图片的坐标，而 region_t 只是原图的一部分，因此需要调整坐标：
    
    # bottomleft 区域（左下）：减去 region_t.shape[3] 以适应 region_t 的相对坐标。
    if best_side == 'bottomleft':
        out_[:, 3] -= region_t.shape[3]
    # bottomright 区域（右下）：既要调整 x 坐标（宽度），又要调整 y 坐标（高度）。
    if best_side == 'bottomright':
        out_[:, 2] -= region_t.shape[2]
        out_[:, 3] -= region_t.shape[3]
    # topright 区域（右上）：只需要调整 x 坐标。
    if best_side == 'topright':
        out_[:, 2] -= region_t.shape[2]  

    # convert to [0, 1]
    # out_[:, 2] 代表目标中心 x 坐标，out_[:, 3] 代表目标中心 y 坐标。
    # out_[:, 4] 代表目标宽度，out_[:, 5] 代表目标高度。
    # 避免目标框超出 region_t
    for jj in range(out_.shape[0]):
        if out_[jj, 2] - out_[jj, 4]/2 < 0:  
            out_[jj, 4] = 2*out_[jj, 2]
        if out_[jj, 2] + out_[jj, 4]/2 > region_t.shape[2]:
            out_[jj, 4] = 2*(region_t.shape[2] - out_[jj, 2])
        if out_[jj, 3] - out_[jj, 5]/2 < 0:  
            out_[jj, 5] = 2*out_[jj, 3]
        if out_[jj, 3] + out_[jj, 5]/2 > region_t.shape[3]:
            out_[jj, 5] = 2*(region_t.shape[3] - out_[jj, 3])

    bboxes_ = out_ # 2400,7
    # 目标框的 x, y, w, h 坐标归一化到 [0, 1] 之间，以适应归一化输入。
    bboxes_[:, 2:6] /= region_t.shape[2]
    # region_t_np = region_t.squeeze(0).cpu().numpy() # squeeze(0) 会移除第一个维度（如果它的大小为1）。(1, 3, 320, 320) -> (3, 320, 320)
    # 假设 region_t 维度为 (batch_size, channels, height, width)，这里取 batch 里的第一张图 (region_t[0])。
    region_t_np = region_t[0].cpu().numpy() # (8, 3, 320, 320) -> (3, 320, 320)
    #   原本的格式是 (C, H, W)（通道优先） 变成 (H, W, C)（通道最后），以适应 transform_ 函数的输入格式。
    region_t_np = np.transpose(region_t_np, (1, 2, 0))
    
    #  移除宽度或高度小于等于 0 的目标框，防止后续计算出错
    bboxes_ = bboxes_[bboxes_[:, 4] > 0]
    bboxes_ = bboxes_[bboxes_[:, 5] > 0]
    # bboxes_.shape = torch.Size([263, 7])

    if bboxes_.shape[0]: # 如果有目标框
        category_ids = [0] * bboxes_.shape[0] # 创建类别 ID（这里默认都为 0）
        transformed = transform_(image=region_t_np, bboxes= bboxes_[:, 2:6], category_ids=category_ids)
        transformed_img =  np.transpose(transformed['image'], (2,0,1)) # 变回 PyTorch 的格式 (C, H, W)，以便继续训练
        bboxes_transformed = transformed['bboxes'] # 增强后的坐标 len():263
        
        # 检查增强后的目标框数量
        if len(bboxes_transformed) != bboxes_.shape[0]:
            print(f"Warning: Number of transformed bboxes ({len(bboxes_transformed)}) does not match original bboxes ({bboxes_.shape[0]}).")
            # 过滤掉无效的目标框
            valid_indices = [i for i, bb in enumerate(bboxes_transformed) if bb is not None]
            bboxes_ = bboxes_[valid_indices]  # 更新 bboxes_

        bboxes_t = [list(bb) for bb in bboxes_transformed] 
        bboxes_t = torch.FloatTensor(bboxes_t)    #  把 list 转换回 PyTorch Tensor [263,4]
        # 确保形状匹配
        if bboxes_t.shape[0] == bboxes_.shape[0]:
            bboxes_t[:, [0, 2]] *= transformed_img.shape[2] # x, w 乘回图片的 宽度，还原真实坐标
            bboxes_t[:, [1, 3]] *= transformed_img.shape[1] # y, h 乘回图片的 高度，还原真实坐标
            bboxes_[:, 2:6] = bboxes_t
        else:
            raise ValueError(f"Shape mismatch: bboxes_t.shape={bboxes_t.shape}, bboxes_.shape={bboxes_.shape}")
        
        # print("bboxes_.shape:", bboxes_.shape) # (,7)
        # print("bboxes_t.shape:", bboxes_t.shape)  # Debugging (,4)
        
        return transformed_img, bboxes_
    else: 
        return region_t.squeeze(0).cpu().numpy(), np.ones((1, 7)) # 如果没有目标框，返回默认值
    