# UDA-YOLO

## Introduction
This is our PyTorch implementation of the paper "[``]()" published in ***   ***.

<div align="center">
    <img src="img/YOLO_DTAD.png" width="1000" alt="UDA-YOLO">
</div>


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
<summary>Test</summary>

```bash
python val.py
```
</details>

<details>
<summary>dual-input</summary>
```shell
/ultralytics/models/yolo/model.py
/ultralytics/models/yolo/detect/__init__.py 
/ultralytics/models/yolo/detect/uda_train.py
/ultralytics/data/uda_build.py  数据集加载  def uda_build_dataloader
/ultralytics/nn/uda_tasks.py  修改模型结构 
/ultralytics/engine/uda_trainer.py 修改训练器
/ultralytics/engine/validator.py  plotting中，主要是为了展示其他损失的变化
```
</details>

