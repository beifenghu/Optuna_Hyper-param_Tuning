# Optuna-hyper-param-tuning-based-on-python-
Optuna 自动超参调整选择，python实现(适用于小模型快速验证寻找最优超参数组合，大模型超参验证请另寻他法(如Ray tune，也是集成了Optuna))


环境:
Ubuntu20.04/Windows11(不定)
pip install optuna
pip install ultralytics  (参考官方文档: https://github.com/ultralytics/ultralytics)
pip install requirement.txt(pytorch框架、python3.10.18、cuda12.6、Nvidia Driver Version: 560.94-3090 24G)

运行:
设置好程序中
1.data_yaml-----你的数据集路径
2.base_model----你的模型路径
3.trials-------寻找参数最优组合试验的次数
4.epochs_per_trial-------每个试验训练的epoch数
5.work_dir--------保存的工作路径(用于保存各个trial结果的)
6.seed--------随机种子，一般默认
7.db_path-----------Optuna的过程记录内置文件路径
8.HYPERPARAM_SPACE-----------超参数搜索空间自定义
9.METRIC_WEIGHTS------------指标权重(用于得出每个trial试验完的最优得分)
10.TPESampler的n_startup_trials和MedianPruner的n_startup_trials---------------在TPESampler的n_startup_trials之后开始加载优化算法计算概率空间，之前全是随机超参组合；在MedianPruner的n_startup_trials之后开始应用剪枝，当此时训练效果比先前trial相同的epoch性能要低很多时，执行截断，不再训练当前trial中剩下的epoch，转而跳转到新的trial继续训练。在MedianPruner的n_startup_trials之前不管性能如何，全部跑完所有trial的epoch。

然后直接python tune_param.py即可。

