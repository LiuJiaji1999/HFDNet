from PIL import Image
from ultralytics.nn.tasks import attempt_load_weights
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

weight = '/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/runs/train/sourceprivate/weights/best.pt'
model = attempt_load_weights(weight, device)
model.info()
for p in model.parameters():
    p.requires_grad_(True)
model.eval()

layer =  [2,4,6,8,9]
target_layers = [model.model[l] for l in layer]
