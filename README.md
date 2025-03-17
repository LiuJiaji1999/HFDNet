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
git clone https://github.com/LiuJiaji1999/udayolo.git 
ultralytics版本为8.2.50,在ultralytics/__init__.py中的__version__有标识.              
pip install -r
    # 3090 单卡
    python: 3.8.18 / 3.8.16
    torch:  1.12.0+cu113 / 1.13.1+cu117
    torchvision: 0.13.0+cu113 / 0.14.1+cu117  
    numpy: 1.22.3
    timm: 0.9.8                 
    mmcv: 2.1.0                
    mmengine: 0.9.0  / 0.10.3    
```

</details>

<details open>
<summary>Train</summary>

```python
python train.py
```
</details>


<details>
<summary>Test</summary>

```bash
python val.py
```
</details>


##### 改为双输入，可实时更新
```shell
/home/lenovo/data/liujiaji/YOLO-DTAD/ultralytics/models/yolo/model.py
/home/lenovo/data/liujiaji/YOLO-DTAD/ultralytics/models/yolo/detect/__init__.py 
/home/lenovo/data/liujiaji/YOLO-DTAD/ultralytics/models/yolo/detect/uda_train.py
/home/lenovo/data/liujiaji/YOLO-DTAD/ultralytics/data/uda_build.py  数据集加载  def uda_build_dataloader
/home/lenovo/data/liujiaji/YOLO-DTAD/ultralytics/nn/uda_tasks.py  修改模型结构 
/home/lenovo/data/liujiaji/YOLO-DTAD/ultralytics/engine/uda_trainer.py 修改训练器
/home/lenovo/data/liujiaji/yolov8/ultralytics-main-8.2.50/ultralytics/engine/validator.py  plotting中，主要是为了展示其他损失的变化
```