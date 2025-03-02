import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def read_pr_data(file_path):
    """
    读取PR曲线数据。
    
    Parameters:
    - file_path: CSV文件路径
    
    Returns:
    - data: DataFrame,包含Precision和Recall列
    """
    data = pd.read_csv(file_path, header=None, names=['Precision', 'Recall'])
    return data

def plot_pr_curve(model_data, linestyle, color_map):
    """
    绘制PR曲线。
    
    Parameters:
    - model_data: 包含各类别PR数据的字典
    - linestyle: 线型
    - color_map: 颜色映射字典
    """
    for category, data in model_data.items():
        plt.plot(data['Recall'], data['Precision'], linestyle=linestyle, color=color_map[category],linewidth=2)

if __name__ == '__main__':
    # 定义模型和类别
    models = ['baseline', 'YOLO-DTADH']
    categories = ['pin-uninstall', 'pin-rust', 'pin-defect', 'Einsu-burn', 'Einsu-defect', 'Einsu-dirty']
    
    # 定义颜色和线型
    color_map = {
        'pin-uninstall': 'b',
        'pin-rust': 'g',
        'pin-defect': 'r',
        'Einsu-burn': 'c',
        'Einsu-defect': 'm',
        'Einsu-dirty': 'y'
    }
    linestyles = { # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
        'baseline': '-',
        'YOLO-DTADH': ':'
    }
    
    # 读取并存储每个模型的PR数据
    all_data = {}
    for model in models:
        model_data = {}
        for category in categories:
            file_path = os.path.join('./PR-curve/'+model, f'{category}.csv')
            model_data[category] = read_pr_data(file_path)
        all_data[model] = model_data

    # 绘制PR曲线
    plt.figure(figsize=(14, 10))  # 调整图形大小
    for model in models:
        plot_pr_curve(all_data[model], linestyles[model], color_map)

    # 创建自定义图例
    handles = []
    labels = []
    for category in categories:
        line = plt.Line2D([0], [0], color=color_map[category], linewidth=2)
        handles.append(line)
        labels.append(category)

    # 显示类别图例
    first_legend = plt.legend(handles, labels, fontsize=12, loc='upper right', title='Categories')
    # first_legend = plt.legend(handles, labels, fontsize=12, loc='upper left', bbox_to_anchor=(1.02, 1), title='Categories')

    # 创建并显示模型图例
    model_handles = [plt.Line2D([0], [0], color='k', linestyle=linestyles[model], linewidth=2) for model in models]
    model_labels = models
    second_legend = plt.legend(model_handles, model_labels, fontsize=12, loc='upper right', bbox_to_anchor=(0.85, 1), title='Models')
    # second_legend = plt.legend(model_handles, model_labels, fontsize=12, loc='upper left', bbox_to_anchor=(1.02, 0.8), title='Models')

    # 添加图例到图表中
    plt.gca().add_artist(first_legend)
    
    # plt.xlabel('Recall', fontsize=16, fontweight='bold')  # 调整x轴标签字体大小和加粗
    plt.xlabel('Recall', fontsize=16)  # 调整x轴标签字体大小和加粗
    plt.ylabel('Precision', fontsize=16)  # 调整y轴标签字体大小和加粗
    plt.xticks(fontsize=14)  # 调整x轴刻度字体大小
    plt.yticks(fontsize=14)  # 调整y轴刻度字体大小
    plt.title('P-R Curve for Different Models and Categories', fontsize=18)  # 调整标题字体大小和加粗
    plt.grid(True, linestyle='--', linewidth=0.5)  # 添加网格线并调整样式
    plt.tight_layout(rect=[0, 0, 0.95, 1])  # 调整布局以防止图例溢出
    plt.savefig('/home/lenovo/data/liujiaji/powerGit/yolov8/image/pr_curve_combined.png')
    # plt.show()  # 显示图表
