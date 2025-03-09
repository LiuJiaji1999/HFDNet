import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8m.yaml')
    # model = YOLO('/home/lenovo/data/liujiaji/yolov8/ultralytics-main/runs/train/exp2/weights/last.pt') # 断点续训
    # model.load('/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/runs/train/baseline/oraclecity/weights/best.pt') # loading pretrain weights
    model.load('yolov8m.pt')
    model.train(data='/home/lenovo/data/liujiaji/powerGit/dayolo/domain/sim10k_to_cityscapes.yaml',
                cache=False,
                imgsz=640,
                epochs=50,
                batch=8, # 32
                close_mosaic=10, 
                workers=8,# 4
                # device='0',
                optimizer='SGD', # using SGD
                patience=0, # set 0 to close earlystop.
                resume=True, # 断点续训,YOLO初始化时选择last.pt
                amp=False, # close amp
                # half=False,
                # fraction=0.2,
                cos_lr = True,
                # project='runs/debug',
                project='runs/train/improve',
                name='sourcesim10k',
                )

        # city_to_foggycity.yaml sourcecity
        # foggycityscapes.yaml oraclefoggy

        # sim10k_to_cityscapes.yaml sourcesim10k
        # cityscapes.yaml oraclecity

        # voc_to_clipart1k.yaml sourcevoc
        # clipart1k.yaml oracleclipart1k

        # privatepower_to_publicpower.yaml sourceprivate
        # publicpower.yaml oraclepublic
