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
source_dataset = CustomDataset(source_path, transform=transform, label_prefix='S')
target_dataset = CustomDataset(target_path, transform=transform, label_prefix='T')

# 合并数据集
combined_dataset = torch.utils.data.ConcatDataset([source_dataset, target_dataset])
st_dataloader = DataLoader(combined_dataset, batch_size=8, shuffle=False, num_workers=4)

# 加载模型
weight = 'runs/train/improve/sourcecity-aptpse-dmm/weights/best.pt'
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
    专业化的t-SNE可视化
    参数:
        features: (N, C) 特征矩阵
        labels: (N,) 类别标签 (格式如'S0', 'T1')
        domains: (N,) 域标签
    """
    # 数据检查
    print("Input features shape:", features.shape)
    print("Sample features:", features[0][:5])  # 打印前5个特征值
    
    # t-SNE降维
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    embeddings = tsne.fit_transform(features)
    print("Embeddings shape:", embeddings.shape)
    
    # 设置样式
    plt.figure(figsize=(14, 10))
    sns.set_style("whitegrid")
    
    
    # 定义颜色和样式
    class_colors = plt.cm.tab10(np.linspace(0, 1, 8))
    style_map = {
        'source': {'marker': 'o', 's': 100, 'alpha': 0.8, 'edgecolor': 'k'},
        'target': {'marker': 's', 's': 100, 'alpha': 0.8, 'edgecolor': 'k'}
    }
    
    # 绘制数据点
    for class_idx in range(8):
        for domain in ['source', 'target']:
            prefix = 'S' if domain == 'source' else 'T'
            mask = (labels == f"{prefix}{class_idx}")
            
            if np.sum(mask) > 0:
                style = style_map[domain]
                plt.scatter(embeddings[mask, 0], embeddings[mask, 1],
                           color=class_colors[class_idx],
                           label=f'Class {class_idx} ({domain})',
                           **style)
    
    # 添加图例和标题
    plt.title('t-SNE Visualization (Debug Mode)', fontsize=16)
    plt.xlabel('t-SNE 1', fontsize=14)
    plt.ylabel('t-SNE 2', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 显示和保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()  # 确保显示图像
    print(f"Visualization saved to {save_path}")
    

# 主流程
if __name__ == "__main__":
    # 1. 提取特征
    features, labels, domains = extract_yolov8_features(model, st_dataloader)
    
    # 3. 可视化
    plot_tsne(features, labels, domains,
             save_path='/home/lenovo/data/liujiaji/powerGit/dayolo/da_tsne.png')