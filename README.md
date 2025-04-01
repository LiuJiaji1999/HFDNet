# HFDet

## Introduction
This is our PyTorch implementation of the paper "[``]()" submitted in ***   ***.

<div align="center">
    <img src="hfdet.png" width="1000" alt="HFDet">
</div>

## Weight
Since github can't upload large files, we uploaded the weights of the four benchmark tasks to the web drive

## <div align="left">Quick Start Examples</div>

<details open>
<summary>Install</summary>

First, clone the project and configure the environment.

```bash
git clone https://github.com/LiuJiaji1999/dgda-yolo.git
ultralytics版本为8.2.50           
NVIDIA GeForce RTX 3090
pip install -r 
    python: 3.8.18
    torch:  1.12.0+cu113
    torchvision: 0.13.0+cu113 
    numpy: 1.22.3   
```
</details>

<details open>
<summary>Train</summary>

```bash
python train.py 
# nohup python train.py > /home/lenovo/data/liujiaji/powerGit/dayolo/logs/improve/c2f.log 2>&1 & tail -f /home/lenovo/data/liujiaji/powerGit/dayolo/logs/improve/c2f.log
```
</details>


<details>
<summary>Test & Detect</summary>

```bash
python val.py
python detect.py
```
</details>

<details>
<summary>dual-input</summary>

```bash
/ultralytics/models/yolo/model.py
/ultralytics/models/yolo/detect/__init__.py 
/ultralytics/models/yolo/detect/uda_train.py
/ultralytics/data/uda_build.py  数据集加载  def uda_build_dataloader
/ultralytics/nn/uda_tasks.py  修改模型结构 
/ultralytics/engine/uda_trainer.py 修改训练器
/ultralytics/engine/validator.py  plotting中，主要是为了展示其他损失的变化
```
</details>

#### Explanation of the file
```bash
# 主要脚本
 train.py ：训练模型的脚本
 val.py ：使用训练好的模型计算指标的脚本
 detect.py ： 推理脚本
# 其他脚本
1. distill.py ： 蒸馏脚本
2. export.py：  导出onnx脚本
3. gap.py: /gap 
4. get_COCO_metrice.py：计算COCO指标的脚本
5. get_FPS.py ：计算模型储存大小、模型推理时间、FPS的脚本
### FPS最严谨来说就是1000(1s)/(preprocess+inference+postprocess),没那么严谨的话就是只除以inference的时间
6. heatmap.py ：生成热力图的脚本
7. main_profile.py ：输出模型和模型每一层的参数,计算量的脚本
8. meanimage.py : 计算均值图像和标准差图像，类 sf-yolo
9. plot_after.py : 有时关闭x，无法保存验证结果图result.png
10. plot_PR : 绘制多个模型的pr结果图
11. plot_result.py：绘制曲线对比图的脚本
12. track.py：跟踪推理的脚本
13. get_model_erf.py ： 绘制模型的有效感受野.
```