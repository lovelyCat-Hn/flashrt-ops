#!/usr/bin/env bash
# G1 脚本统一启动器：自带 SDK 运行库路径 + flash_pyrt311 环境。
# 用法: ~/holy/run.sh ~/holy/scripts/inference/run_g1_loop.py --exec
#   （脚本路径后的参数原样透传；新开 shell 可直接用，无需定义变量）
exec env LD_LIBRARY_PATH=/data/galbot/lib PYTHONPATH=/data/galbot/lib \
    /home/galbot/miniforge3/envs/flash_pyrt311/bin/python "$@"
