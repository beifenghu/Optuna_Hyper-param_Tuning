#!/usr/bin/env python3
# coding: utf-8
"""
基于Optuna的YOLO超参数优化脚本（优化版）

优化点：
1. 使用Optuna的标准Trial接口进行参数采样（参考官方示例）
2. 简化参数空间定义，使用更直观的字典结构
3. 优化剪枝策略，使用MedianPruner（参考官方文档）
4. 改进试验恢复机制，确保状态一致性
5. 添加更详细的日志和进度跟踪
6. 优化存储和映射管理，减少冗余代码
"""

import optuna
import subprocess
import yaml
import os
from pathlib import Path
import pandas as pd
import sys
import signal
import json
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import argparse
from datetime import datetime
import shutil
import time
import traceback
from optuna.trial import Trial as OptunaTrial
from optuna.trial import TrialState
import optuna.trial as ot_trial

# ===========================
# 配置
# ===========================
data_yaml = "chengdu.yaml"
base_model = "yolo11s.pt"
TRIALS_TO_RUN = 25
EPOCHS_PER_TRIAL = 300
work_dir = Path("optuna_temp")
work_dir.mkdir(exist_ok=True)
trial_status_file = work_dir / "trial_status.json"
seed = 42
db_path = work_dir / "study.db"
USED_PARAMS_FILE = work_dir / "used_params.json"

# ===========================
# 超参数空间（参考官方示例简化定义）
# ===========================
HYPERPARAM_SPACE = {
    "lr0": (1e-5, 0.1, "log"),
    "lrf": (0.01, 0.5),
    "imgsz": [1024],
    "close_mosaic": [120],
    "optimizer": ["SGD", "Adam"],
    "momentum": (0.8, 0.98),
    "weight_decay": (1e-5, 1e-3, "log"),
    "box": (0.5, 5.0),
    "cls": (0.5, 3.0),
    "dfl": (0.5, 3.0),
    "kobj": (0.5, 3.0),
    "hsv_h": (0.0, 0.1),
    "hsv_s": (0.0, 0.9),
    "hsv_v": (0.0, 0.9),
    "fliplr": (0.0, 0.9),
    "mosaic": (0.5, 1.0),
    "mixup": (0.0, 0.3),
    "batch": [16, 24],
    "cos_lr": [True, False],
}

# 评估指标权重
METRIC_WEIGHTS = {
    "metrics/mAP50(B)": 0.4,
    "metrics/mAP50-95(B)": 0.3,
    "metrics/precision(B)": 0.15,
    "metrics/recall(B)": 0.15
}

# 全局映射 fs_trial -> optuna trial number
FS_TO_OPTUNA = {}

# ===========================
# 工具函数
# ===========================
def safe_unlink(path: Path, max_attempts=5, delay=1):
    """安全删除文件，处理权限问题"""
    if not path.exists():
        return True
    for attempt in range(max_attempts):
        try:
            path.unlink()
            return True
        except PermissionError:
            if attempt < max_attempts - 1:
                time.sleep(delay)
                delay *= 2
            else:
                print(f"警告：无法删除文件 {path}, 可能被其他进程占用")
                return False
        except Exception as e:
            print(f"删除文件 {path} 时出错: {e}")
            return False
    return False

def load_used_params():
    """加载已使用的参数组合"""
    if USED_PARAMS_FILE.exists():
        try:
            with open(USED_PARAMS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_used_params(params_list):
    """保存已使用的参数组合"""
    with open(USED_PARAMS_FILE, 'w') as f:
        json.dump(params_list, f, indent=2)

def get_existing_trial_numbers():
    """获取已存在的试验编号"""
    existing = []
    for item in work_dir.glob("trial_*"):
        if item.is_dir():
            try:
                num = int(item.name.split("_")[1])
                existing.append(num)
            except Exception:
                pass
    return sorted(existing)

def load_trial_status():
    """加载试验状态"""
    if trial_status_file.exists():
        try:
            with open(trial_status_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_trial_status(status):
    """保存试验状态"""
    with open(trial_status_file, 'w') as f:
        json.dump(status, f, indent=2)

def update_trial_status(trial_num, epoch, status="running"):
    """更新试验状态"""
    trial_status = load_trial_status()
    trial_status[str(trial_num)] = {
        "status": status,
        "last_epoch": epoch,
        "last_updated": datetime.now().isoformat()
    }
    save_trial_status(trial_status)

def get_incomplete_trials_from_status():
    """从状态文件中获取未完成的试验"""
    trial_status = load_trial_status()
    existing = get_existing_trial_numbers()
    nums = [int(trial_num) for trial_num, info in trial_status.items() 
            if info["status"] in ["running", "interrupted"] and int(trial_num) in existing]
    nums.sort()
    return nums

def calculate_best_score(trial_dir: Path):
    """计算试验的最佳得分（参考官方示例的评估方式）"""
    best_score = 0.0
    train_dir = trial_dir / "train"
    if not train_dir.exists():
        train_dir = trial_dir
    results_file = train_dir / "results.csv"
    if results_file.exists():
        try:
            df = pd.read_csv(results_file)
            for _, row in df.iterrows():
                score = 0.0
                for m, w in METRIC_WEIGHTS.items():
                    if m in row:
                        score += float(row[m]) * w
                if score > best_score:
                    best_score = score
        except Exception as e:
            print(f"Warning: failed to calculate best score: {e}")
    return best_score

def get_last_epoch(trial_dir: Path):
    """获取试验的最后epoch和检查点"""
    train_dir = trial_dir / "train"
    if not train_dir.exists():
        train_dir = trial_dir
    results_file = train_dir / "results.csv"
    last_epoch = 0
    last_ckpt = train_dir / "weights" / "last.pt"
    if results_file.exists():
        try:
            df = pd.read_csv(results_file)
            if not df.empty and "epoch" in df.columns:
                last_epoch = int(df.iloc[-1]["epoch"])
        except Exception:
            pass
    return last_epoch, last_ckpt if last_ckpt.exists() else None

# ===========================
# 参数采样（参考官方示例）
# ===========================
def sample_params(trial):
    """使用Optuna Trial对象采样参数（参考官方示例的define_model函数）"""
    params = {}
    
    # 学习率
    if "lr0" in HYPERPARAM_SPACE:
        low, high, *log = HYPERPARAM_SPACE["lr0"]
        params["lr0"] = trial.suggest_float("lr0", low, high, log=len(log) > 0 and log[0] == "log")
    
    # 学习率衰减
    if "lrf" in HYPERPARAM_SPACE:
        low, high = HYPERPARAM_SPACE["lrf"][:2]
        params["lrf"] = trial.suggest_float("lrf", low, high)
    
    # 图像尺寸（固定值）
    if "imgsz" in HYPERPARAM_SPACE:
        params["imgsz"] = trial.suggest_categorical("imgsz", HYPERPARAM_SPACE["imgsz"])
    
    # 关闭mosaic的epoch
    if "close_mosaic" in HYPERPARAM_SPACE:
        params["close_mosaic"] = trial.suggest_categorical("close_mosaic", HYPERPARAM_SPACE["close_mosaic"])
    
    # 优化器
    if "optimizer" in HYPERPARAM_SPACE:
        params["optimizer"] = trial.suggest_categorical("optimizer", HYPERPARAM_SPACE["optimizer"])
    
    # 动量
    if "momentum" in HYPERPARAM_SPACE:
        low, high = HYPERPARAM_SPACE["momentum"][:2]
        params["momentum"] = trial.suggest_float("momentum", low, high)
    
    # 权重衰减
    if "weight_decay" in HYPERPARAM_SPACE:
        low, high, *log = HYPERPARAM_SPACE["weight_decay"]
        params["weight_decay"] = trial.suggest_float("weight_decay", low, high, log=len(log) > 0 and log[0] == "log")
    
    # 损失函数权重
    for loss_param in ["box", "cls", "dfl", "kobj"]:
        if loss_param in HYPERPARAM_SPACE:
            low, high = HYPERPARAM_SPACE[loss_param][:2]
            params[loss_param] = trial.suggest_float(loss_param, low, high)
    
    # 数据增强参数
    for aug_param in ["hsv_h", "hsv_s", "hsv_v", "fliplr", "mosaic", "mixup"]:
        if aug_param in HYPERPARAM_SPACE:
            low, high = HYPERPARAM_SPACE[aug_param][:2]
            params[aug_param] = trial.suggest_float(aug_param, low, high)
    
    # 批次大小
    if "batch" in HYPERPARAM_SPACE:
        params["batch"] = trial.suggest_categorical("batch", HYPERPARAM_SPACE["batch"])
    
    # 学习率调度器
    if "cos_lr" in HYPERPARAM_SPACE:
        params["cos_lr"] = trial.suggest_categorical("cos_lr", HYPERPARAM_SPACE["cos_lr"])
    
    return params

# ===========================
# 试验映射管理
# ===========================
def refresh_fs_to_optuna(study):
    """刷新文件系统试验到Optuna试验的映射"""
    global FS_TO_OPTUNA
    FS_TO_OPTUNA = {}
    try:
        for t in study.trials:
            try:
                fa = t.user_attrs.get("fs_trial", None)
            except Exception:
                fa = None
            if fa is not None:
                try:
                    FS_TO_OPTUNA[int(fa)] = t.number
                except Exception:
                    pass
    except Exception as e:
        print(f"refresh_fs_to_optuna 出错: {e}")
    return FS_TO_OPTUNA

def finalize_fs_trial(study, fs_num, value, state=TrialState.COMPLETE):
    """完成文件系统试验并更新Optuna试验状态（参考官方示例的完成逻辑）"""
    try:
        # 查找对应的Optuna试验
        optuna_num = None
        for trial in study.trials:
            if trial.user_attrs.get("fs_trial") == fs_num:
                optuna_num = trial.number
                break
        
        if optuna_num is None:
            print(f"警告：未找到fs_trial {fs_num}对应的Optuna试验")
            return False
        
        # 使用study.tell完成试验
        study.tell(optuna_num, value, state=state)
        print(f"成功完成试验 fs_trial {fs_num} -> optuna_trial {optuna_num}, 值: {value}, 状态: {state}")
        return True
    except Exception as e:
        print(f"完成试验时出错: {e}")
        return False

# ===========================
# 目标函数（参考官方示例的objective函数）
# ===========================
def objective(trial):
    """目标函数：运行YOLO训练并返回评估分数"""
    # 采样参数
    params = sample_params(trial)
    
    # 创建试验目录
    trial_dir = work_dir / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存参数到文件
    params_file = trial_dir / "params.yaml"
    with open(params_file, 'w') as f:
        yaml.dump(params, f)
    
    # 设置用户属性以映射文件系统试验
    trial.set_user_attr("fs_trial", trial.number)
    
    # 构建训练命令
    cmd = [
        "yolo", "train",
        f"data={data_yaml}",
        f"model={base_model}",
        f"epochs={EPOCHS_PER_TRIAL}",
        f"project={trial_dir}",
        "exist_ok=True",
        "verbose=False"
    ]
    
    # 添加超参数
    for k, v in params.items():
        if isinstance(v, bool):
            cmd.append(f"{k}={'True' if v else 'False'}")
        else:
            cmd.append(f"{k}={v}")
    
    # 运行训练
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    try:
        best_score = 0.0
        for line in process.stdout:
            line_clean = line.strip()
            
            # 解析训练输出
            if "Epoch" in line_clean and "metrics/" in line_clean:
                try:
                    # 解析指标（简化版，实际应根据YOLO输出格式调整）
                    if "mAP50" in line_clean:
                        mAP50 = float(line_clean.split("mAP50:")[1].split()[0])
                        mAP5095 = float(line_clean.split("mAP50-95:")[1].split()[0])
                        precision = float(line_clean.split("precision:")[1].split()[0])
                        recall = float(line_clean.split("recall:")[1].split()[0])
                        
                        # 计算加权得分
                        score = (mAP50 * METRIC_WEIGHTS["metrics/mAP50(B)"] +
                                 mAP5095 * METRIC_WEIGHTS["metrics/mAP50-95(B)"] +
                                 precision * METRIC_WEIGHTS["metrics/precision(B)"] +
                                 recall * METRIC_WEIGHTS["metrics/recall(B)"])
                        
                        # 更新最佳得分
                        if score > best_score:
                            best_score = score
                        
                        # 报告中间值（参考官方示例）
                        trial.report(score, step=int(line_clean.split("Epoch")[1].split("/")[0]))
                        
                        # 检查是否应该剪枝（参考官方示例）
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
                            
                except (IndexError, ValueError) as e:
                    # 解析失败，继续下一行
                    continue
                    
    except optuna.exceptions.TrialPruned:
        # 试验被剪枝，终止训练进程
        process.terminate()
        raise
        
    finally:
        # 确保进程终止
        if process.poll() is None:
            process.terminate()
    
    # 等待进程结束
    process.wait()
    
    # 计算最终得分
    final_score = calculate_best_score(trial_dir)
    
    # 更新试验状态
    update_trial_status(trial.number, EPOCHS_PER_TRIAL, "completed")
    
    return final_score

# ===========================
# 信号处理
# ===========================
def signal_handler(sig, frame):
    """处理中断信号"""
    trial_status = load_trial_status()
    for k, info in trial_status.items():
        if info["status"] == "running":
            info["status"] = "interrupted"
    save_trial_status(trial_status)
    print("Ctrl+C detected. Trial statuses saved. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ===========================
# 主流程
# ===========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO超参数优化")
    parser.add_argument("--trials", type=int, default=TRIALS_TO_RUN, help="要运行的试验数量")
    parser.add_argument("--epochs", type=int, default=EPOCHS_PER_TRIAL, help="每个试验的epoch数")
    parser.add_argument("--prune", action="store_true", help="启用剪枝")
    parser.add_argument("--resume", action="store_true", help="恢复未完成的试验")
    args = parser.parse_args()

    # 更新配置
    TRIALS_TO_RUN = args.trials
    EPOCHS_PER_TRIAL = args.epochs

    # 创建或加载研究
    storage_url = f"sqlite:///{db_path}"
    study = optuna.create_study(
        study_name="yolo_optuna",
        direction="maximize",
        storage=storage_url,
        sampler=TPESampler(seed=seed, n_startup_trials=10),
        pruner=MedianPruner(n_startup_trials=10) if args.prune else optuna.pruners.NopPruner(),
        load_if_exists=True
    )

    # 恢复未完成的试验（如果指定）
    if args.resume:
        incomplete_trials = []
        for trial in study.trials:
            if trial.state == TrialState.RUNNING or trial.state == TrialState.PRUNED:
                incomplete_trials.append(trial)
        
        if incomplete_trials:
            print(f"找到 {len(incomplete_trials)} 个未完成的试验，尝试恢复...")
            for trial in incomplete_trials:
                try:
                    # 重新运行目标函数
                    objective(trial)
                except Exception as e:
                    print(f"恢复试验 {trial.number} 时出错: {e}")

    # 运行新试验
    study.optimize(objective, n_trials=TRIALS_TO_RUN, show_progress_bar=True)

    # 输出最佳试验结果（参考官方示例）
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len([t for t in study.trials if t.state == TrialState.PRUNED]))
    print("  Number of complete trials: ", len([t for t in study.trials if t.state == TrialState.COMPLETE]))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
