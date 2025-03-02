# import os
# import numpy as np
# import torch
# import torchvision.transforms as transforms
# import torchvision.models as models
# from torch.utils.data import Dataset, DataLoader
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import cv2


# import warnings
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO
# from ultralytics.nn.tasks import attempt_load_weights

# # 固定随机种子
# np.random.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# # 自定义数据集类
# class CustomDataset(Dataset):
#     def __init__(self, root_dir, transform=None,label_prefix=''):
#         self.root_dir = root_dir
#         self.image_dir = os.path.join(root_dir, 'img')
#         self.label_dir = os.path.join(root_dir, 'txt')
#         self.transform = transform
#         self.label_prefix = label_prefix  # 每个数据集的唯一前缀
#         self.image_paths = sorted([os.path.join(self.image_dir, fname) for fname in os.listdir(self.image_dir)])
#         self.label_paths = sorted([os.path.join(self.label_dir, fname) for fname in os.listdir(self.label_dir)])

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         label_path = self.label_paths[idx]
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = transforms.ToPILImage()(image)
#         if self.transform:
#             image = self.transform(image)
#         with open(label_path, 'r') as f:
#             label = f.readline().strip().split(' ')[0]  # 读取标签中的id部分
#         label = f"{self.label_prefix}{label}"  # 添加前缀
#         return image, label

# # 数据预处理
# transform = transforms.Compose([
#     transforms.Resize((640, 640)),
#     transforms.ToTensor(),
# ])

# # 加载不同数据集
# dataset1 = CustomDataset(root_dir='/home/lenovo/data/liujiaji/Datasets/Einsulator/defect/', transform=transform,label_prefix='E1-')
# dataset2 = CustomDataset(root_dir='/home/lenovo/data/liujiaji/Datasets/CPLID/Defective_Insulators/', transform=transform,label_prefix='C1-')
# dataset3 = CustomDataset(root_dir='/home/lenovo/data/liujiaji/Datasets/VPMBGI/', transform=transform,label_prefix='B1-')
# dataset4 = CustomDataset(root_dir='/home/lenovo/data/liujiaji/Datasets/Einsulator/burn/', transform=transform,label_prefix='E2-')
# dataset5 = CustomDataset(root_dir='/home/lenovo/data/liujiaji/Datasets/IDID/train/', transform=transform,label_prefix='I1-')

# # 合并数据集
# combined_dataset = torch.utils.data.ConcatDataset([dataset1, dataset2, dataset3, dataset4, dataset5])
# dataloader = DataLoader(combined_dataset, batch_size=8, shuffle=True, num_workers=4)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 其他hub可下载的预训练模型，v5,6,7
# # model = torch.hub.load('ultralytics/yolov5', 'yolov5x' ,pretrained=True).to(device)
# # model.eval()

# # v8 参照heatmap
# # weight = '/home/lenovo/data/liujiaji/yolov8/ultralytics-main/yolov8m.pt'
# weight = 'runs/train/exp2/weights/best.pt'
# model = attempt_load_weights(weight, device)
# # print(model)
# model.eval()


# # 特征提取
# features = []
# labels = []
# for images, lbls in tqdm(dataloader, desc='Running the model inference'):
#     images = images.to(device)

#     # 其他hub可下载的预训练模型，v5,6,7
#     # output = model(images) # torch.Size([8, 25200, 85])
 
#     output = model(images)[0] # torch.Size([8, 84, 8400])
#     features.append(output.cpu().detach().numpy())
#     labels.extend(lbls)

# print('1',len(features))  # 180

# features = np.concatenate(features, axis=0)
# print('2',len(features))  # 1440
# print('2',features.shape)  # (1440, 84, 8400)

# features = np.array(features).reshape(len(features), -1)
# print('3',len(features))  # 1440
# print('3',features.shape)  # (1440, 705600)


# # 打印特征和标签的长度
# print(f'Number of features: {len(features)}') # 1440
# print(f'Number of labels: {len(labels)}') # 1440

# # 确保特征和标签数量一致
# assert len(features) == len(labels), "Features and labels length mismatch!"


# # t-SNE降维
# tsne = TSNE(n_components=2, random_state=42)
# tsne_results = tsne.fit_transform(features) 
# print(tsne_results.shape) #(1440,2)

# # 映射标签到新名称
# label_mapping = {
#     'E1-1': 'insulator-defect',
#     'B1-4': 'VPMBGI-defect',
#     'C1-defect': 'CPLID-defect',
#     'I1-3': 'IDID-broken',

#     'E2-0': 'insulator-burn',
#     'I1-4': 'IDID-flashover',

# }

# # 将标签映射到新标签名
# mapped_labels = [label_mapping.get(str(label), str(label)) for label in labels]

# # 选择要可视化的标签
# # visualize_labels = ['insulator-defect', 'VPMBGI-defect','CPLID-defect','IDID-broken']  # 仅可视化标签为 
# visualize_labels = ['insulator-burn', 'IDID-flashover']

# # 可视化
# def scale_to_01_range(x):
#     value_range = (np.max(x) - np.min(x))
#     starts_from_zero = x - np.min(x)
#     return starts_from_zero / value_range

# tx = tsne_results[:, 0]
# ty = tsne_results[:, 1]

# tx = scale_to_01_range(tx)
# ty = scale_to_01_range(ty)

# # 不同数据集标签的不同样式
# styles = {
#     'insulator-defect': ('o', 'lightsalmon'),
#     'CPLID-defect':('+','cyan'),
#     'VPMBGI-defect':('*','seagreen'),
#     'IDID-broken': ('^', 'skyblue'),

#     'insulator-burn': ('s', 'green'),
#     'IDID-flashover': ('D', 'pink'),
    

# }

# # 绘制2D点，每个点的颜色与类标签对应
# fig, ax = plt.subplots()
# unique_labels = np.unique(mapped_labels)
# # print(unique_labels)
# colors_per_class = {label: plt.cm.tab10(i % 10) for i, label in enumerate(unique_labels)}

# for label in unique_labels:
#     if label in visualize_labels: 
#         indices = [i for i, l in enumerate(mapped_labels) if l == label] # i是标签索引，l是标签值
#         current_tx = np.take(tx, indices)
#         current_ty = np.take(ty, indices)
#         color = colors_per_class[label]
#         # new_label = label_mapping.get(label, label)  # 映射到新标签名
#         # # print(f"Original label: {label}, Mapped label: {new_label}")
#         # ax.scatter(current_tx, current_ty, c=[color], label=label, alpha=0.9, edgecolors='k', linewidth=0.9)

#         # marker, color = styles.get(label, ('o', 'black'))  # 默认样式为黑色圆点
#         # ax.scatter(current_tx, current_ty, c=color, marker=marker, label=label, alpha=0.9, edgecolors='black', linewidth=0.8)

# # ax.legend(loc='best')
# # plt.title('t-SNE visualization of the dataset')
# # plt.savefig('/home/lenovo/data/liujiaji/powerGit/dataset/tsne-defect-ours-1.jpg')

# numpy == 1.23.5

import perceptual_manifold_geometry as pmg
import numpy as np

data = np.random.rand(100, 3)
print(data)

overall_curvature = pmg.curvatures(data, k=15, pca_components=8, curvature_type='PCA')
print(f"Overall curvature: {overall_curvature}")

gaussian_curvature = pmg.curvatures(data, k=15, pca_components=8, curvature_type='gaussian')
print(f"Gaussian curvature: {gaussian_curvature}")

mean_curvature = pmg.curvatures(data, k=15, pca_components=8, curvature_type='mean')
print(f"Mean curvature: {mean_curvature}")

# Calculate volume and density
volume, density = pmg.calculate_volume(data)
print(f"Volume: {volume}, Density: {density}")

# Estimate holes and plot persistence diagrams
num_holes, total_size, mean_size, density_holes = pmg.estimate_holes_ripser(data, threshold=0.1, Persistence_diagrams=True)
print(f"Number of Holes: {num_holes}, Total Size: {total_size}, Mean Size: {mean_size}, Density Holes: {density_holes}")

# Estimate intrinsic dimension
intrinsic_dim = pmg.estimate_intrinsic_dimension(data, method='TLE')
print(f"Intrinsic Dimension: {intrinsic_dim}")

# Estimate nonconvexity
nonconvexity = pmg.estimate_nonconvexity(data)
print(f"Nonconvexity: {nonconvexity}")