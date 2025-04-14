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

def extract_features(model, dataloader, hook_index):
    features = []
    def hook(module, input, output):
        pooled_feat = F.adaptive_avg_pool2d(output, (1, 1))
        features.append(pooled_feat.view(-1).cpu().numpy())
    handle = model.model[hook_index].register_forward_hook(hook)

    model.eval()
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            model(images)
    handle.remove()
    return np.array(features)

def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff + np.trace(sigma1 + sigma2 - 2 * covmean)

# **数据加载**
# c2f
# source_path = '/home/lenovo/data/liujiaji/DA-Datasets/CityScapes/yolov5_format'
# target_path = '/home/lenovo/data/liujiaji/DA-Datasets/CityScapesFoggy/yolov5_format'
# daod_weight = 'runs/train/improve/sourcecity-aptpse-dmm/weights/best.pt'
# source_weight = 'runs/train/baseline/sourcecity/weights/best.pt'

# s2c
# source_path = '/home/lenovo/data/liujiaji/DA-Datasets/Sim10k'
# target_path = '/home/lenovo/data/liujiaji/DA-Datasets/CityScapes/yolov5_format_car_class'
# daod_weight = 'runs/train/improve/sourcesim10k-aptpse-dmm/weights/best.pt'
# source_weight = 'runs/train/baseline/sourcesim10k/weights/best.pt'

# # v2c
source_path = '/home/lenovo/data/liujiaji/DA-Datasets/VOC/train/VOCdevkit/VOC2007/yolov5_format'
target_path = '/home/lenovo/data/liujiaji/DA-Datasets/clipart/yolov5_format'
daod_weight = 'runs/train/improve/sourcevoc-aptpse-dmm/weights/best.pt'
source_weight = 'runs/train/baseline/sourcevoc/weights/best.pt'

# pu2pr
# source_path = '/home/lenovo/data/liujiaji/Datasets/pupower'
# target_path = '/home/lenovo/data/liujiaji/Datasets/prpower'
# daod_weight = 'runs/train/improve/sourcepu-aptpse-dmm/weights/best.pt'
# source_weight = 'runs/train/baseline/sourcepu/weights/best.pt'

# 数据集加载（保持不变）
source_dataset = CustomDataset(source_path, transform=transform)
target_dataset = CustomDataset(target_path, transform=transform)
source_loader = DataLoader(source_dataset, batch_size=1, shuffle=False, num_workers=4)
target_loader = DataLoader(target_dataset, batch_size=1, shuffle=False, num_workers=4)

# 加载模型并提取特征（图像级，使用 model[9] 即 SPPF）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_source = attempt_load_weights(source_weight, device)
model_adapted = attempt_load_weights(daod_weight, device)

features_src_src = extract_features(model_source, source_loader, hook_index=9)
features_src_tgt = extract_features(model_source, target_loader, hook_index=9)
features_adapt_src = extract_features(model_adapted, source_loader, hook_index=9)
features_adapt_tgt = extract_features(model_adapted, target_loader, hook_index=9)

# 计算 FID
def get_fid(f1, f2):
    mu1, sigma1 = np.mean(f1, axis=0), np.cov(f1, rowvar=False)
    mu2, sigma2 = np.mean(f2, axis=0), np.cov(f2, rowvar=False)
    return calculate_fid(mu1, sigma1, mu2, sigma2)

fid_baseline = get_fid(features_src_src, features_src_tgt)
fid_adapted = get_fid(features_adapt_src, features_adapt_tgt)

## ================= PCA 降维 =================
all_feats = np.vstack([features_src_src, features_adapt_src, features_src_tgt, features_adapt_tgt])
pca = PCA(n_components=2)
feats_2d = pca.fit_transform(all_feats)

N = len(source_dataset)
M = len(target_dataset)

# ================= 可视化 =================
plt.figure(figsize=(10, 8))
plt.scatter(feats_2d[:N, 0], feats_2d[:N, 1], label='Source-Only (Source)', c='blue', marker='o', alpha=0.6)
plt.scatter(feats_2d[N:2*N, 0], feats_2d[N:2*N, 1], label='HFDet (Source)', c='cyan', marker='x', alpha=0.6)
plt.scatter(feats_2d[2*N:2*N+M, 0], feats_2d[2*N:2*N+M, 1], label='Source-Only (Target)', c='red', marker='o', alpha=0.6)
plt.scatter(feats_2d[2*N+M:, 0], feats_2d[2*N+M:, 1], label='HFDet (Target)', c='orange', marker='x', alpha=0.6)
plt.legend()

leg = plt.legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)
print(f"FID\nBaseline: {fid_baseline:.2f}\nHFDet: {fid_adapted:.2f}")

# 获取数据范围
x_max, y_max = feats_2d[:, 0].max(), feats_2d[:, 1].max()
x_min, y_min = feats_2d[:, 0].min(), feats_2d[:, 1].min()

# # 设置 FID 注释位置：右上角，稍微偏移一些，避免遮挡边界
# plt.text(
#     x=x_max - 0.05 * (x_max - x_min),
#     y=y_max - 0.05 * (y_max - y_min),
#     s=f"FID\nBaseline: {fid_baseline:.2f}\nHFDet: {fid_adapted:.2f}",
#     fontsize=12,
#     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
# )
plt.title("PCA Visualization")
plt.tight_layout()
plt.savefig('./gap/pca/pca-v2c-compare.png', dpi=300)
