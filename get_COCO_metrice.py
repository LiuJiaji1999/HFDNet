import warnings
warnings.filterwarnings('ignore')
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tidecv import TIDE, datasets

def parse_opt():
    parser = argparse.ArgumentParser()
    # 数据集转化后生成的json文件路径,,, 用/ultralytics-main/dataset/yolo2coco.py生成的数据集 json
    parser.add_argument('--anno_json', type=str, default='/home/lenovo/data/liujiaji/yolov8/powerdata/test.json', help='training model path')
    # # val后生成的预测json文件路径
    # parser.add_argument('--pred_json', type=str, default='/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/runs/test/exp/predictions.json', help='data yaml path')
    parser.add_argument('--pred_json', type=str, default='/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.1.9/runs/test/exp81/predictions.json', help='data yaml path')
    
    # val后生成的预测json文件路径
    # parser.add_argument('--pred_json', type=str, default='/home/lenovo/data/liujiaji/yolov9/runs/test/exp2/best_predictions.json', help='data yaml path')
    
    return parser.parse_known_args()[0]

if __name__ == '__main__':
    opt = parse_opt()
    anno_json = opt.anno_json
    pred_json = opt.pred_json
    
    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    tide = TIDE()
    tide.evaluate_range(datasets.COCO(anno_json), datasets.COCOResult(pred_json), mode=TIDE.BOX)
    tide.summarize()
    tide.plot(out_dir='result')