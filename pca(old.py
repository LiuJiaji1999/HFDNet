import torch
import numpy as np
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from ultralytics.nn.tasks import attempt_load_weights
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import cv2
from scipy.linalg import sqrtm  # 用于计算矩阵平方根

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images/val')
        self.image_paths = sorted([os.path.join(self.image_dir, fname) 
                                   for fname in os.listdir(self.image_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transforms.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)
        return image

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# **数据加载**
# c2f
source_path = '/home/lenovo/data/liujiaji/DA-Datasets/CityScapes/yolov5_format'
target_path = '/home/lenovo/data/liujiaji/DA-Datasets/CityScapesFoggy/yolov5_format'
weight = 'runs/train/improve/sourcecity-aptpse-dmm/weights/best.pt'
# weight = 'runs/train/baseline/sourcecity/weights/best.pt'

# s2c
# source_path = '/home/lenovo/data/liujiaji/DA-Datasets/Sim10k'
# target_path = '/home/lenovo/data/liujiaji/DA-Datasets/CityScapes/yolov5_format_car_class'
# # weight = 'runs/train/improve/sourcesim10k-aptpse-dmm/weights/best.pt'
# weight = 'runs/train/baseline/sourcesim10k/weights/best.pt'

# # # v2c
# source_path = '/home/lenovo/data/liujiaji/DA-Datasets/VOC/train/VOCdevkit/VOC2007/yolov5_format'
# target_path = '/home/lenovo/data/liujiaji/DA-Datasets/clipart/yolov5_format'
# weight = 'runs/train/improve/sourcevoc-aptpse-dmm/weights/best.pt'
# # weight = 'runs/train/baseline/sourcevoc/weights/best.pt'

# pu2pr
# source_path = '/home/lenovo/data/liujiaji/Datasets/pupower'
# target_path = '/home/lenovo/data/liujiaji/Datasets/prpower'
# # weight = 'runs/train/improve/sourcepu-aptpse-dmm/weights/best.pt'
# weight = 'runs/train/baseline/sourcepu/weights/best.pt'

source_dataset = CustomDataset(source_path, transform=transform)
target_dataset = CustomDataset(target_path, transform=transform)

source_loader = DataLoader(source_dataset, batch_size=1, shuffle=False, num_workers=4)
target_loader = DataLoader(target_dataset, batch_size=1, shuffle=False, num_workers=4)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# weight = 'yolov8m.pt'  # 替换为你的模型权重路径
model = attempt_load_weights(weight, device).eval()

# **记录特征的字典**
image_level_features = {}
proposal_level_features = {}

# **Hook 函数**
def image_hook(module, input, output):
    pooled_feat = F.adaptive_avg_pool2d(output, (1, 1))  # 自适应平均池化
    image_level_features['features'].append(pooled_feat.view(-1).cpu().numpy())

def proposal_hook(module, input, output):
    pooled_feat = F.adaptive_avg_pool2d(output, (1, 1))  # 类似 ROI 特征
    proposal_level_features['features'].append(pooled_feat.view(-1).cpu().numpy())

# **注册 Hook**
image_level_features['features'] = []
proposal_level_features['features'] = []
backbone_hook_handle = model.model[9].register_forward_hook(image_hook)  # SPPF
roi_heads_hook_handle = model.model[10].register_forward_hook(proposal_hook)  # 预测头


# **特征提取**
with torch.no_grad():
    for images in source_loader:
        images = images.to(device)
        model(images)

    for images in target_loader:
        images = images.to(device)
        model(images)

# **移除 Hook**
backbone_hook_handle.remove()
roi_heads_hook_handle.remove()

# **PCA 处理**
source_features = np.array(image_level_features['features'][:len(source_dataset)])
target_features = np.array(image_level_features['features'][len(source_dataset):])

combined_features = np.vstack([source_features, target_features])

pca = PCA(n_components=2)
features_2d = pca.fit_transform(combined_features)

# **可视化**
plt.figure(figsize=(8, 6))
plt.scatter(features_2d[:len(source_features), 0], features_2d[:len(source_features), 1], c='blue', label='Source', alpha=0.6)
plt.scatter(features_2d[len(source_features):, 0], features_2d[len(source_features):, 1], c='red', label='Target', alpha=0.6)

leg = plt.legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)

# 计算 Frechet Inception Distance (FID)
def calculate_fid(mu1, sigma1, mu2, sigma2):
    """ 计算 Frechet Inception Distance (FID) """
    diff = np.sum((mu1 - mu2) ** 2)  # 均值差异的 L2 范数
    covmean = sqrtm(sigma1 @ sigma2)  # 计算协方差矩阵的平方根
    if np.iscomplexobj(covmean):  # 处理可能的小数误差导致的复数问题
        covmean = covmean.real  
    return diff + np.trace(sigma1 + sigma2 - 2 * covmean)

# 计算均值和协方差
mu_s, sigma_s = np.mean(source_features, axis=0), np.cov(source_features, rowvar=False)
mu_t, sigma_t = np.mean(target_features, axis=0), np.cov(target_features, rowvar=False)

fid_score = calculate_fid(mu_s, sigma_s, mu_t, sigma_t)
print(f"Frechet Distance (FID) Score: {fid_score:.4f}")

# plt.title(f"PCA Visualization (FID={fid_score:.2f})(Exp. Var. {pca.explained_variance_ratio_.sum():.2f})")
plt.title(f"PCA Visualization (FID={fid_score:.2f})")
plt.savefig('./gap/pca/pca-c2f.png', dpi=300, bbox_inches='tight')
