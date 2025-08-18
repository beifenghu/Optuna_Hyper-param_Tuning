import optuna
import subprocess
import yaml
import os
import re
from pathlib import Path
import pandas as pd
import sys
import signal
import json
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import argparse
from datetime import datetime

# ===========================
# 配置路径和训练参数
# ===========================
data_yaml = ".yaml"
base_model = "yolo11s.pt"
trials = 50
epochs_per_trial = 50
work_dir = Path("optuna_temp")
work_dir.mkdir(exist_ok=True)
trial_status_file = work_dir / "trial_status.json"
seed = 42
db_path = work_dir / "study.db"

# ===========================
# 超参数空间
# ===========================
HYPERPARAM_SPACE = {
    "lr0": {"low": 1e-5, "high": 0.1, "log": True, "type": "float"},
    "lrf": {"low": 0.01, "high": 0.5, "log": False, "type": "float"},
    "imgsz": {"choices": [1024], "type": "categorical"},
    "close_mosaic": {"choices": [120], "type": "categorical"},
    "optimizer": {"choices": ["SGD", "Adam"], "type": "categorical"},
    "momentum": {"low": 0.8, "high": 0.98, "log": False, "type": "float"},
    "weight_decay": {"low": 1e-5, "high": 1e-3, "log": True, "type": "float"},
    "box": {"low": 0.5, "high": 5.0, "log": False, "type": "float"},
    "cls": {"low": 0.5, "high": 3.0, "log": False, "type": "float"},
    "dfl": {"low": 0.5, "high": 3.0, "log": False, "type": "float"},
    "kobj": {"low": 0.5, "high": 3.0, "log": False, "type": "float"},
    "hsv_h": {"low": 0.0, "high": 0.1, "log": False, "type": "float"},
    "hsv_s": {"low": 0.0, "high": 0.9, "log": False, "type": "float"},
    "hsv_v": {"low": 0.0, "high": 0.9, "log": False, "type": "float"},
    "fliplr": {"low": 0.0, "high": 0.9, "log": False, "type": "float"},
    "mosaic": {"low": 0.5, "high": 1.0, "log": False, "type": "float"},
    "mixup": {"low": 0.0, "high": 0.3, "log": False, "type": "float"},
    "batch": {"choices": [16, 24], "type": "categorical"},
    "cos_lr": {"choices": [True, False], "type": "categorical"},
}

METRIC_WEIGHTS = {
    "metrics/mAP50(B)": 0.4,
    "metrics/mAP50-95(B)": 0.3,
    "metrics/precision(B)": 0.15,
    "metrics/recall(B)": 0.15
}

# ===========================
# Trial 状态管理
# ===========================
def load_trial_status():
    if trial_status_file.exists():
        with open(trial_status_file, 'r') as f:
            return json.load(f)
    return {}

def save_trial_status(status):
    with open(trial_status_file, 'w') as f:
        json.dump(status, f, indent=2)

def update_trial_status(trial_num, epoch, status="running"):
    trial_status = load_trial_status()
    trial_status[str(trial_num)] = {
        "status": status,
        "last_epoch": epoch,
        "last_updated": datetime.now().isoformat()
    }
    save_trial_status(trial_status)

def get_incomplete_trials():
    trial_status = load_trial_status()
    return [int(trial_num) for trial_num, info in trial_status.items()
            if info["status"] in ["running", "interrupted"]]

# ===========================
# 参数采样
# ===========================
def sample_params(trial):
    params = {}
    for k, v in HYPERPARAM_SPACE.items():
        if v["type"] == "float":
            params[k] = trial.suggest_float(k, v["low"], v["high"], log=v["log"])
        elif v["type"] == "categorical":
            params[k] = trial.suggest_categorical(k, v["choices"])
    return params

# ===========================
# 获取 epoch & checkpoint
# ===========================
def get_last_epoch(trial_dir: Path):
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
# 计算最佳分数
# ===========================
def calculate_best_score(trial_dir: Path):
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

# ===========================
# 目标函数
# ===========================
def objective(trial, resume_trial_dir=None, fixed_number=None):
    trial_num = fixed_number if fixed_number is not None else trial.number
    trial_dir = work_dir / f"trial_{trial_num}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    params_file = trial_dir / "params.yaml"

    # ===== 修改点：区分 resume 和新 trial =====
    if resume_trial_dir is not None:
        # 恢复训练：必须用之前保存的参数
        if params_file.exists():
            with open(params_file, 'r') as f:
                params = yaml.safe_load(f)
        else:
            raise RuntimeError(f"恢复 trial {trial_num} 失败，缺少 params.yaml")
    else:
        # 新的 trial：强制重新采样参数并覆盖保存
        params = sample_params(trial)
        with open(params_file, 'w') as f:
            yaml.dump(params, f)

    # ===== 继续原逻辑 =====
    last_completed_epoch, last_ckpt = get_last_epoch(trial_dir)
    start_epoch = last_completed_epoch

    if start_epoch >= epochs_per_trial:
        best_score = calculate_best_score(trial_dir)
        print(f"Trial {trial_num} 已完成，最优加权得分: {best_score:.4f}")
        return best_score

    remaining_epochs = epochs_per_trial - start_epoch

    cmd = [
        "yolo", "train",
        f"data={data_yaml}",
        f"epochs={remaining_epochs}",
        f"project={trial_dir}",
        "exist_ok=True",
        "verbose=False"
    ]
    if last_ckpt and start_epoch > 0:
        cmd.append(f"model={last_ckpt}")
        cmd.append(f"resume=True")
    else:
        cmd.append(f"model={base_model}")

    for k, v in params.items():
        if isinstance(v, bool):
            cmd.append(f"{k}={'True' if v else 'False'}")
        else:
            cmd.append(f"{k}={v}")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            line_clean = line.strip()
            if "Epoch" in line_clean:
                train_dir = trial_dir / "train"
                if not train_dir.exists():
                    train_dir = trial_dir
                results_file = train_dir / "results.csv"
                if results_file.exists():
                    try:
                        df = pd.read_csv(results_file)
                        if not df.empty:
                            row = df.iloc[-1]
                            precision = float(row["metrics/precision(B)"])
                            recall = float(row["metrics/recall(B)"])
                            mAP50 = float(row["metrics/mAP50(B)"])
                            mAP5095 = float(row["metrics/mAP50-95(B)"])
                            current_epoch = int(row["epoch"])
                            score = (mAP50 * METRIC_WEIGHTS["metrics/mAP50(B)"] +
                                     mAP5095 * METRIC_WEIGHTS["metrics/mAP50-95(B)"] +
                                     precision * METRIC_WEIGHTS["metrics/precision(B)"] +
                                     recall * METRIC_WEIGHTS["metrics/recall(B)"])

                            print(f"Trial {trial_num} | Epoch {current_epoch}/{epochs_per_trial} | "
                                  f"precision={precision:.3f}, recall={recall:.3f}, "
                                  f"mAP50={mAP50:.3f}, mAP50-95={mAP5095:.3f}")

                            update_trial_status(trial_num, current_epoch)

                            trial.report(score, current_epoch)
                            if trial.should_prune():
                                print(f"Trial {trial_num} 在 Epoch {current_epoch} 被剪枝")
                                update_trial_status(trial_num, current_epoch, "pruned")
                                process.terminate()
                                raise optuna.TrialPruned()
                    except Exception as e:
                        print(f"读取 results.csv 出错: {e}")
        process.wait()

        train_dir = trial_dir / "train"
        if not train_dir.exists():
            train_dir = trial_dir
        results_file = train_dir / "results.csv"
        if results_file.exists():
            try:
                df = pd.read_csv(results_file)
                if not df.empty:
                    row = df.iloc[-1]
                    precision = float(row["metrics/precision(B)"])
                    recall = float(row["metrics/recall(B)"])
                    mAP50 = float(row["metrics/mAP50(B)"])
                    mAP5095 = float(row["metrics/mAP50-95(B)"])
                    current_epoch = int(row["epoch"])
                    print(f"Trial {trial_num} | Epoch {current_epoch}/{epochs_per_trial} | "
                          f"precision={precision:.3f}, recall={recall:.3f}, "
                          f"mAP50={mAP50:.3f}, mAP50-95={mAP5095:.3f}")
                    update_trial_status(trial_num, current_epoch)
            except Exception as e:
                print(f"读取 results.csv 出错: {e}")

        if process.returncode != 0:
            update_trial_status(trial_num, start_epoch, "failed")
            best_score = calculate_best_score(trial_dir)
            return best_score
    except KeyboardInterrupt:
        process.terminate()
        update_trial_status(trial_num, start_epoch, "interrupted")
        raise

    best_score = calculate_best_score(trial_dir)
    update_trial_status(trial_num, epochs_per_trial, "completed")
    print(f"Trial {trial_num} 完成，最优加权得分: {best_score:.4f}")
    return best_score


# ===========================
# 信号处理
# ===========================
def signal_handler(sig, frame):
    trial_status = load_trial_status()
    for info in trial_status.values():
        if info["status"] == "running":
            info["status"] = "interrupted"
    save_trial_status(trial_status)
    print("Ctrl+C detected. Trial statuses saved. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ===========================
# 主函数
# ===========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=trials)
    parser.add_argument("--epochs", type=int, default=epochs_per_trial)
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-trial", type=int)
    args = parser.parse_args()

    trials = args.trials
    epochs_per_trial = args.epochs
    sampler = TPESampler(seed=seed, n_startup_trials=30)
    pruner = MedianPruner(n_startup_trials=30) if args.prune else optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name="yolo_optuna",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{db_path}",
        load_if_exists=True
    )

    if args.resume_trial is not None:
        trial_dir = work_dir / f"trial_{args.resume_trial}"
        if trial_dir.exists():
            study.optimize(lambda t: objective(t, resume_trial_dir=trial_dir, fixed_number=args.resume_trial), n_trials=1)
        sys.exit(0)

    if args.resume:
        # 先恢复所有未完成的 trial
            for trial_num in get_incomplete_trials():
                trial_dir = work_dir / f"trial_{trial_num}"
                if trial_dir.exists():
                    study.optimize(
                        lambda t: objective(t, resume_trial_dir=trial_dir, fixed_number=trial_num),
                        n_trials=1
                    )

            # 重新确认哪些 trial 已经完成
            completed_nums = {
                t.number for t in study.get_trials() 
                if t.state == optuna.trial.TrialState.COMPLETE
            }

            # 接着跑剩下的新 trial，直到达到 --trials 指定的总数
            for trial_num in range(trials):
                if trial_num not in completed_nums:
                    study.optimize(
                        lambda t: objective(t, fixed_number=trial_num),
                        n_trials=1
                    )

    else:
        study.optimize(objective, n_trials=trials)

    print("=== All trials completed ===")
    print(f"Best trial: {study.best_trial.number}, score={study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")
