#!/bin/bash

# 定义训练命令列表（按顺序执行）
commands=(
    "python feature_gap.py"
    "python feature_gap_1.py"
    "python feature_gap_2.py"
)

# 设置最大重试次数
MAX_RETRIES=3

# 遍历并执行每个训练命令
for cmd in "${commands[@]}"; do
    log_file="${cmd// /_}.log"  # 生成日志文件名
    retries=0  # 记录当前命令的重试次数

    while [ $retries -lt $MAX_RETRIES ]; do
        echo "Executing: $cmd (Attempt $((retries + 1))/$MAX_RETRIES)..."
        
        # 后台执行命令，并让其不中断
        nohup $cmd > "$log_file" 2>&1 &
        pid=$!  # 获取进程 ID
        echo "Started: $cmd (PID: $pid), logging to $log_file"

        # 等待进程完成
        wait $pid
        exit_code=$?

        if [ $exit_code -eq 0 ]; then
            echo "Success: $cmd completed successfully."
            break  # 退出重试循环，执行下一个命令
        else
            echo "Error: $cmd failed with exit code $exit_code."
            echo "Check log: $log_file"
            ((retries++))  # 增加重试次数
            sleep 10  # 等待 10 秒后重试
        fi
    done

    if [ $retries -eq $MAX_RETRIES ]; then
        echo "Max retries reached for $cmd. Skipping to next command."
    fi
done

echo "All commands have been executed."
