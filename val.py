import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 验证参数官方详解链接：https://docs.ultralytics.com/modes/val/#usage-examples:~:text=of%20each%20category-,Arguments%20for%20YOLO%20Model%20Validation,-When%20validating%20YOLO

if __name__ == '__main__':
    model = YOLO('/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/runs/train/improve/sourcecity7/weights/best.pt')
    model.val(data='/home/lenovo/data/liujiaji/powerGit/dayolo/domain/privatepower_to_publicpower.yaml',
              split='val',
              imgsz=800,
              batch=8,
              # iou=0.7,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val/improve',
              name='sourcecity',
              )
    # city_to_foggycity.yaml sourcecity
    # foggycityscapes.yaml oraclefoggy

    # sim10k_to_cityscapes.yaml sourcesim10k
    # cityscapes.yaml oraclecity

    # voc_to_clipart1k.yaml sourcevoc
    # clipart1k.yaml oracleclipart1k

    # privatepower_to_publicpower.yaml sourceprivate
    # publicpower.yaml oraclepublic