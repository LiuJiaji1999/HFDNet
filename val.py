import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 验证参数官方详解链接：https://docs.ultralytics.com/modes/val/#usage-examples:~:text=of%20each%20category-,Arguments%20for%20YOLO%20Model%20Validation,-When%20validating%20YOLO

if __name__ == '__main__':
    model = YOLO('/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/runs/train/baseline/sourcepu/weights/last.pt')
    model.val(data='/home/lenovo/data/liujiaji/powerGit/dayolo/domain/pupower_to_prpower.yaml',
              split='val',
              imgsz=640,
              batch=8,
              # iou=0.7,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val/baseline',
              name='sourcepu',
              )
    
    # city_to_foggycity.yaml sourcecity
    # foggycityscapes.yaml oraclefoggy

    # sim10k_to_cityscapes.yaml sourcesim10k
    # cityscapes.yaml oraclecity

    # voc_to_clipart1k.yaml sourcevoc
    # clipart1k.yaml oracleclipart1k

    # publicpower_to_privatepower.yaml sourcepublic
    # privatepower.yaml oracleprivate

    # pupower_to_prpower.yaml sourcepu
    # prpower.yaml oraclepr