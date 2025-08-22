# Optuna Hyper-parameter Tuning for Ultralytics YOLO

![Optuna Logo](https://github.com/beifenghu/Optuna_Hyper-param_Tuning/blob/main/111.png)  <!-- 替换为你的图片路径或URL -->

## 项目简介
基于 Optuna 的自动超参数调整方案，适用于 Ultralytics YOLO 模型的小规模快速验证。通过设置搜索空间和评估指标权重，自动寻找最优超参数组合。

> ⚠️ **重要提示**：本方案仅适用于小模型快速验证。对于大模型超参数验证，请改用 Ray Tune（已集成 Optuna）等分布式方案。

---

## 环境创建(不定)
- 操作系统：Ubuntu 20.04 / Windows 11
- Python 3.10.18
- PyTorch 2.x (CUDA 12.6)
- NVIDIA 驱动版本：560.94
- GPU：NVIDIA 3090 24G

---

## 安装依赖

- pip install optuna 
- pip install ultralytics  # 官方文档: https://github.com/ultralytics/ultralytics
- pip install -r requirements.txt  # 包含 PyTorch 和 CUDA 依赖

---
## 程序处理流程图总览
![Optuna Logo]( https://github.com/beifenghu/Optuna_Hyper-param_Tuning/blob/main/Optuna.png)
## 运行

设置好程序中

- data_yaml----- 你的数据集路径

- base_model---- 你的模型路径

- TRIALS_TO_RUN------- 进行多少次试验

- EPOCHS_PER_TRIAL------- 每个试验训练的 epoch 数

- work_dir-------- 保存的工作路径 (用于保存各个 trial 结果的)

- seed-------- 随机种子，一般默认

- db_path-----------Optuna 的过程记录内置文件路径

- HYPERPARAM_SPACE----------- 超参数搜索空间自定义

- METRIC_WEIGHTS------------ 指标权重 (用于得出每个 trial 试验完的最优得分)

- TPESampler 的 n_startup_trials 和 MedianPruner 的 n_startup_trials:
- 在 TPESampler 的 n_startup_trials 之后开始加载优化算法计算概率空间，之前全是随机超参组合；
- 在 MedianPruner 的 n_startup_trials 之后开始应用剪枝，当此时训练效果比先前 trial 相同的 epoch 性能要低很多时，执行截断，不再训练当前 trial 中剩下的 epoch，转而跳转到新的 trial 继续训练。在 MedianPruner 的 n_startup_trials 之前不管性能如何，全部跑完所有 trial 的 epoch。

然后直接
- python tune_param.py --prune  #用于初次启动调优代码，--prune视个人情况开启关闭
- python tune_param.py --prune --resume  #对中断trial进行寻找并重新训练，直到所有trial全部实验完成

## optuna-dashboard参数可视化

终端输入

- optuna-dashboard sqlite:///optuna_temp/study.db --host 0.0.0.0 --port 8080

然后在浏览器中打开

- http://localhost:8080

即可查看训练过程及Optuna应用统计学方法得出的参数重要性等一些图表，辅助调优
