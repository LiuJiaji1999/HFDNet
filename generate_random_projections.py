import numpy as np
import torch
import pickle
import os


def save_variable(file_name, data):
    # 以二进制写入模式（wb）打开文件。
    with open(file_name, "wb") as file:
        # 将 data 对象序列化为二进制格式，并写入到文件中。
        pickle.dump(data, file)

def load_variable(file_name):
    with open(file_name, "rb") as file:
        data = pickle.load(file)
    return data

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='plot correlation between domain gap and detection accuracy')
    parser.add_argument('--save-dir', default="", metavar='FILE', help='path to save projections');
    parser.add_argument('--num-projs', default=10, help='number of projections to be generated');
    args = parser.parse_args()
    return args


def rand_projections(dim, num_projections=1000, device="cuda"):
    # 生成形状为 (num_projections, dim) 的随机张量，服从标准正态分布。
    projections = torch.randn((num_projections, dim)).to(device)
    # 归一化：将每个向量除以其自身的L2范数（欧几里得范数），使每个向量的长度为1。
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True)) 
     # 返回生成的标准化投影向量张量
    return projections

def main():
    args = parse_args();
    print("\nCommand Line Args:", args);
    print("\n");

    os.makedirs(args.save_dir, exist_ok=True);
    for dim in [4177920, 8355840, 16711680, 24343434 ,24343434]:

        projections = rand_projections(dim, num_projections=args.num_projs);
        projections = projections.detach().cpu().numpy();

        save_variable(file_name=args.save_dir + "/projections_n" + str(args.num_projs) + "_d" + str(dim), data=projections);

    print("Done!!!");

if __name__ == "__main__":
    main();