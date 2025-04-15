import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# import itertools 
# import random
# # 定义搜索空间
# gamma_weights = [0.01, 0.05, 0.1, 0.5, 1.0, 1.5 ]  # gamma_weight 的候选值
# alpha_weights = [0.01, 0.05, 0.1, 0.5, 1.0, 1.5]  # alpha_weight 的候选值
# lambda_weights = [0.01, 0.05, 0.1, 0.5, 1.0, 1.5]  # lambda_weight 的候选值

# # # 生成所有可能的组合 网格搜索
# # param_grid = list(itertools.product(gamma_weights, alpha_weights, lambda_weights ))
# # # 生成所有可能的组合，并过滤掉不满足 alpha_weight < lambda_weight 的组合
# # param_grid = [
# #     (gamma, alpha, lambda_ )
# #     for gamma, alpha, lambda_ in itertools.product(gamma_weights, alpha_weights, lambda_weights)
# #     if alpha < lambda_  # 确保 alpha_weight < lambda_weight
# # ]
# # 随机采样次数 随机搜索
# n_iter = 3  # 随机采样 20 次
# # 随机采样参数组合
# param_samples = [
#     (
#         random.choice(gamma_weights),
#         random.choice(alpha_weights),
#         random.choice(lambda_weights),
#     )
#     for _ in range(n_iter)
# ]

# best_mAP = 0  # 记录最佳验证集性能
# best_params = None  # 记录最佳参数组合
    
# for gamma_weight, alpha_weight, lambda_weight in param_samples:
    
#     model = YOLO('ultralytics/cfg/models/v8/yolov8m.yaml')
#     # model = YOLO('/home/lenovo/data/liujiaji/yolov8/ultralytics-main/runs/train/exp2/weights/last.pt') # 断点续训
#     # 伪标签使用的 源域 pre-trained weight
#     # model.load('/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/runs/train/baseline/sourcecity/weights/best.pt') # loading pretrain weights
#     # COCO pre-trained weight
#     model.load('yolov8m.pt')
#     result = model.train(data='/home/lenovo/data/liujiaji/powerGit/dayolo/domain/city_to_foggycity.yaml',
#                 cache=False,
#                 imgsz=640,
#                 epochs=1,
#                 batch=8, # 32
#                 close_mosaic=10, 
#                 workers=8,# 4
#                 # device='0',
#                 optimizer='SGD', # using SGD
#                 patience=0, # set 0 to close earlystop.
#                 resume=True, # 断点续训,YOLO初始化时选择last.pt
#                 amp=False, # close amp
#                 # half=False,
#                 # fraction=0.2,
#                 cos_lr = True,
#                 project='runs/debug',
#                 # project='runs/train/improve',
#                 name='sourcecity',
#                 # mixup = 1.0,
#                 # mosaic = 0.0,

#                 # 设置自定义损失权重
#                 gamma_weight = gamma_weight,
#                 alpha_weight = alpha_weight,
#                 lambda_weight = lambda_weight
#                 )
#     # # 验证集性能
#     if result.results_dict['metrics/mAP50(B)'] > best_mAP :
#         best_mAP = result.results_dict['metrics/mAP50(B)']
#         best_params = (gamma_weight, alpha_weight, lambda_weight)
#     print(f"Params: gamma={gamma_weight}, alpha={alpha_weight}, lambda={lambda_weight}, Val mAP50: {val_mAP}")

# # 输出最优结果
# print(f"Best mAP50: {best_mAP}")
# print(f"Best parameters (gamma, alpha, lambda): {best_params}")

        # city_to_foggycity.yaml sourcecity
        # foggycityscapes.yaml oraclefoggy
        
        # sim10k_to_cityscapes.yaml sourcesim10k
        # cityscapes.yaml oraclecity

        # voc_to_clipart1k.yaml sourcevoc
        # clipart1k.yaml oracleclipart1k
        
        # pupower_to_prpower.yaml sourcepu
        # prpower.yaml oraclepr

# # 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings
if __name__ == '__main__':
    # model = YOLO('ultralytics/cfg/models/v8/yolov8m.yaml')
    # model = YOLO('/home/lenovo/data/liujiaji/yolov8/ultralytics-main/runs/train/exp2/weights/last.pt') # 断点续训
    # 域适应会使用 源域 pre-trained weight
    # model.load('/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/runs/train/baseline/sourcepu2/weights/best.pt') # loading pretrain weights
    # COCO pre-trained weight
    # model.load('yolov8m.pt')
    
    model = YOLO('ultralytics/cfg/models/v5/yolov5m.yaml')
    # model.load('yolov5mu.pt')
    model.load('/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/runs/train/v5/sourcevoc/weights/best.pt') # loading pretrain weights
    
    result = model.train(data='/home/lenovo/data/liujiaji/powerGit/dayolo/domain/voc_to_clipart1k.yaml',
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
                project='runs/train/v5',
                name = 'v2c',
                # mixup = 1.0,
                )
