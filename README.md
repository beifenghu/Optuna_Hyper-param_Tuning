# Optuna Hyper-parameter Tuning for Ultralytics YOLO

![Optuna Logo](C://Users//Administrator//Desktop\111.png)  <!-- 替换为你的图片路径或URL -->

## 项目简介
基于 Optuna 的自动超参数调整方案，适用于 Ultralytics YOLO 模型的小规模快速验证。通过设置搜索空间和评估指标权重，自动寻找最优超参数组合。

> ⚠️ **重要提示**：本方案仅适用于小模型快速验证。对于大模型超参数验证，请改用 Ray Tune（已集成 Optuna）等分布式方案。

---

## 环境要求
- 操作系统：Ubuntu 20.04 / Windows 11
- Python 3.10.18
- PyTorch 2.x (CUDA 12.6)
- NVIDIA 驱动版本：560.94
- GPU：NVIDIA 3090 24G

---

## 安装依赖
```bash
pip install optuna
pip install ultralytics  # 官方文档: https://github.com/ultralytics/ultralytics
pip install -r requirements.txt  # 包含 PyTorch 和 CUDA 依赖

## 运行

## 设置好程序中
- data_yaml-----你的数据集路径
- base_model----你的模型路径
- trials-------寻找参数最优组合试验的次数
- epochs_per_trial-------每个试验训练的epoch数
- work_dir--------保存的工作路径(用于保存各个trial结果的)
- seed--------随机种子，一般默认
- db_path-----------Optuna的过程记录内置文件路径
- HYPERPARAM_SPACE-----------超参数搜索空间自定义
- METRIC_WEIGHTS------------指标权重(用于得出每个trial试验完的最优得分)
- TPESampler的n_startup_trials和MedianPruner的n_startup_trials---------------在TPESampler的n_startup_trials之后开始加载优化算法计算概率空间，之前全是随机超参组合；在MedianPruner的n_startup_trials之后开始应用剪枝，当此时训练效果比先前trial相同的epoch性能要低很多时，执行截断，不再训练当前trial中剩下的epoch，转而跳转到新的trial继续训练。在MedianPruner的n_startup_trials之前不管性能如何，全部跑完所有trial的epoch。

## 然后直接
- python tune_param.py
## 即可。

