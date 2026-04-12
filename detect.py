import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from pathlib import Path
# 推理参数官方详解链接：https://docs.ultralytics.com/modes/predict/#inference-sources:~:text=of%20Results%20objects-,Inference%20Arguments,-model.predict()
       
# city_to_foggycity.yaml  sim10k_to_cityscapes.yaml voc_to_clipart1k.yaml publicpower_to_privatepower
# source-only: baseline/sourcecity            /sourcesim10k              /sourcevoc               /sourcepublic 
# ours:       improve/sourcecity-aptpse-dmm  /sourcesim10k-aptpse-dmm    /sourcevoc-aptpse-dmm    /sourcepublic-aptpse-dmm2
# oracle:      baseline/oraclefoggy          /oraclecity                 /oracleclipart1k         /oracleprivate sourceprivate

# /home/lenovo/data/liujiaji/Datasets-DA/CityScapesFoggy/yolov5_format/images/val
# /home/lenovo/data/liujiaji/Datasets-DA/CityScapes/yolov5_format_car_class/images/val
# /home/lenovo/data/liujiaji/Datasets-DA/clipart/yolov5_format/images/val
# /home/lenovo/data/liujiaji/Datasets-Power/prpower/images/val 

if __name__ == '__main__':
    model = YOLO('/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/runs/train/baseline/sourceprpin/weights/best.pt') # select your model.pt path
    # 图片目录
#     source_dir = Path('/home/lenovo/data/liujiaji/Datasets-Power/privatepower-s-t/source/images/test')
#     # 支持的图片格式
#     exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.PNG']
#     # 收集所有图片
#     img_paths = []
#     for ext in exts:
#         img_paths.extend(source_dir.glob(ext))
#     # 排序（保证顺序一致）
#     img_paths = sorted(img_paths)
#     # 只取前200张
#     img_paths = img_paths[:200]
#     print(f"Total selected images: {len(img_paths)}")
    # 转成字符串列表（YOLO需要）
#     img_paths = [str(p) for p in img_paths]
    model.predict(
                  # source= img_paths,
                  source='/home/lenovo/data/liujiaji/Datasets-Power/privatepower-s-t/source/images/test', 
                  # target='/home/lenovo/data/liujiaji/Datasets/prpower/images/val',
                  imgsz=960,
                  project='runs/detect',
                  name='prpower_to_prpower-pin/oracle',
                  save=True, # result save
                  conf=0.6,
                  # stream=True,
                  # iou=0.7,
                  # agnostic_nms=True,
                  # visualize=True, # visualize model features maps
                  # line_width=8, # line width of the bounding boxes
                  show_conf=False, # do not show prediction confidence
                  show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_conf=True
                  # save_crop=True, # save cropped images with results
                )

   ##  可以对结果进行后处理！！！
#     for i in model.predict(source='dataset/images/test',
#                   imgsz=640,
#                   project='runs/detect',
#                   name='exp',
#                   save=True,):
#             print(i)
'''              
boxes: ultralytics.engine.results.Boxes object
keypoints: None
masks: None
names: {0: 'pin-defect', 1: 'pin-rust', 2: 'pin-uninstall', 3: 'Einsu-burn', 4: 'Einsu-defect', 5: 'Einsu-dirty'}
obb: None
orig_img: array([[[129, 114, 112],
        [126, 111, 109],
        [125, 109, 110],
        ...,
        [245, 225, 224],
        [244, 223, 221],
        [236, 215, 213]],
  ...,
        [ 75,  69,  62],
        [ 87,  80,  77],
        [ 85,  75,  75]]], dtype=uint8)
orig_shape: (864, 1152)
path: '/home/lenovo/data/liujiaji/yolov8/ultralytics-main/dataset/images/test/3424.jpg'
probs: None
save_dir: 'runs/detect/exp3'
speed: {'preprocess': 3.0732154846191406, 'inference': 12.158632278442383, 'postprocess': 4.642248153686523}
ultralytics.engine.results.Results object with attributes:   
'''