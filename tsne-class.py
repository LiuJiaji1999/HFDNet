import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics.nn.tasks import attempt_load_weights
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import cv2
import torch.nn as nn
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, label_prefix=''):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images/val')
        self.label_dir = os.path.join(root_dir, 'labels/val')
        self.transform = transform
        self.label_prefix = label_prefix
        self.image_paths = sorted([os.path.join(self.image_dir, fname) 
                                 for fname in os.listdir(self.image_dir)])
        self.label_paths = sorted([os.path.join(self.label_dir, fname) 
                                  for fname in os.listdir(self.label_dir)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transforms.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)
        with open(label_path, 'r') as f:
            label = f.readline().strip().split(' ')[0]
        label = f"{self.label_prefix}{label}"
        return image, label

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import cv2
import torch.nn as nn
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 自定义数据集类保持不变...

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# 加载数据
source_path = '/home/lenovo/data/liujiaji/DA-Datasets/CityScapes/yolov5_format'
target_path = '/home/lenovo/data/liujiaji/DA-Datasets/CityScapesFoggy/yolov5_format'

# source_path = '/home/lenovo/data/liujiaji/DA-Datasets/Sim10k'
# target_path = '/home/lenovo/data/liujiaji/DA-Datasets/CityScapes/yolov5_format_car_class'

source_dataset = CustomDataset(source_path, transform=transform, label_prefix='S')
target_dataset = CustomDataset(target_path, transform=transform, label_prefix='T')

# 合并数据集
combined_dataset = torch.utils.data.ConcatDataset([source_dataset, target_dataset])
st_dataloader = DataLoader(combined_dataset, batch_size=8, shuffle=False, num_workers=4)

# 加载模型
weight = 'runs/train/improve/sourcecity-aptpse-dmm/weights/best.pt'
# weight = 'runs/train/baseline/sourcecity/weights/best.pt'
model = attempt_load_weights(weight, device).eval()

def extract_yolov8_features(model, dataloader):
    """
    提取YOLOv8模型最后的特征输出
    返回:
        features: (N, C) 经过平均池化后的特征
        labels: (N,) 样本标签
        domains: (N,) 域标签('source'/'target')
    """
    features = []
    labels = []
    domains = []
    
    with torch.no_grad():
        for images, lbls in tqdm(dataloader, desc='Extracting features'):
            images = images.to(device)
            outputs = model(images)
            
            # 调试输出检查
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]  # 取第一个输出
            
            # 确保是3D特征 [batch, channels, features]
            if outputs.dim() == 3:
                pooled = outputs.mean(dim=-1)  # [batch, channels]
            else:
                pooled = outputs  # 已经是2D特征
                
            features.append(pooled.cpu().numpy())
            labels.extend(lbls)
            domains.extend(['source' if lbl.startswith('S') else 'target' for lbl in lbls])
    
    features = np.concatenate(features)
    print(f"Final features shape: {features.shape}")
    return features, np.array(labels), np.array(domains)


def plot_tsne(features, labels, domains, save_path='tsne_visualization.png'):
    """
    简洁大气的t-SNE可视化
    参数:
        features: (N, C) 特征矩阵
        labels: (N,) 类别标签 (格式如'S0', 'T1')
        domains: (N,) 域标签
    """
    # t-SNE降维
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    embeddings = tsne.fit_transform(features)
    
    # 设置样式
    plt.figure(figsize=(12, 8))
    sns.set_style("white")
    
    # 定义8个类别的颜色 (使用更柔和的颜色)
    class_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
    ]
    
    # 定义域样式
    markers = {'source': 'o', 'target': 's'}  # 源域圆形，目标域方形
    
    # 绘制数据点
    for class_idx in range(8):
        for domain in ['source', 'target']:
            prefix = 'S' if domain == 'source' else 'T'
            mask = (labels == f"{prefix}{class_idx}")
            
            if np.sum(mask) > 0:
                plt.scatter(embeddings[mask, 0], embeddings[mask, 1],
                           c=class_colors[class_idx],
                           marker=markers[domain],
                           s=60,
                           alpha=0.7,
                           edgecolor='w',
                           linewidth=0.5,
                           label=f'C{class_idx}({domain[0].upper()})')
    
    # 添加图例和标题
    plt.title('Feature Distribution Visualization', fontsize=14, pad=12)
    # plt.xlabel('Dimension 1', fontsize=12)
    # plt.ylabel('Dimension 2', fontsize=12)
    
    # 简化图例
    handles, labels_legend = plt.gca().get_legend_handles_labels()
    # 去重
    by_label = dict(zip(labels_legend, handles))
    plt.legend(by_label.values(), by_label.keys(),
               frameon=True,
               fontsize=10,
               bbox_to_anchor=(1.05, 1),
               borderaxespad=0.)
    
    # 美化图形
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_facecolor('#f5f5f5')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")



# 主流程
if __name__ == "__main__":
    # 1. 提取特征
    features, labels, domains = extract_yolov8_features(model, st_dataloader)
    
    # 3. 可视化
    plot_tsne(features, labels, domains,
             save_path='/home/lenovo/data/liujiaji/powerGit/dayolo/tsne-c2f.png')