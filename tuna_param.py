#!/usr/bin/env python3
# coding: utf-8
"""
Optuna 恢复逻辑改进脚本（A1a：未完成导入为 RUNNING，但一次仅允许导入/继续一个 RUNNING）
 - clean_invalid_trials() 会把确实完成的 fs trial 导入为 COMPLETE；
   把最多一个未完成的 fs trial 导入为 RUNNING（其余未完成的跳过导入，以避免并行恢复冲突）。
 - resume 时：优先复用 DB 中的 RUNNING trial（OptunaTrial(study, id)），否则 enqueue->ask。
 - 已完成 trial 从文件系统导入时使用 create_trial + add_trial（官方建议的低级 API）。
注意：请在运行前备份现有 DB 文件（optuna_temp/study.db）。
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
import random
import math
from optuna.trial import Trial as OptunaTrial
from optuna.trial import TrialState
from optuna.distributions import FloatDistribution, CategoricalDistribution
import optuna.trial as ot_trial

# ===========================
# 配置路径和训练参数（可根据需要修改）
# ===========================
data_yaml = ".yaml"
base_model = "yolo11s.pt"
TRIALS_TO_RUN = 50
EPOCHS_PER_TRIAL = 50
work_dir = Path("optuna_temp")
work_dir.mkdir(exist_ok=True)
trial_status_file = work_dir / "trial_status.json"
seed = 42
db_path = work_dir / "study.db"
USED_PARAMS_FILE = work_dir / "used_params.json"

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

# 全局映射 fs_trial -> optuna trial number（会在加载 study 后构建）
FS_TO_OPTUNA = {}

# ===========================
# 工具函数（基本保持你的实现）
# ===========================
def safe_unlink(path: Path, max_attempts=5, delay=1):
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
                print(f"警告：无法删除文件 {path}，可能被其他进程占用")
                return False
        except Exception as e:
            print(f"删除文件 {path} 时出错: {e}")
            return False

def load_used_params():
    if USED_PARAMS_FILE.exists():
        try:
            with open(USED_PARAMS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_used_params(params_list):
    with open(USED_PARAMS_FILE, 'w') as f:
        json.dump(params_list, f, indent=2)

def is_param_used(params, used_params, tolerance=1e-6):
    for used in used_params:
        match = True
        for key in params:
            if key not in used:
                match = False
                break
            if isinstance(params[key], float) and isinstance(used[key], float):
                if abs(params[key] - used[key]) > tolerance:
                    match = False
                    break
            else:
                if params[key] != used[key]:
                    match = False
                    break
        if match:
            return True
    return False

def get_existing_trial_numbers():
    existing = []
    for item in work_dir.glob("trial_*"):
        if item.is_dir():
            try:
                num = int(item.name.split("_")[1])
                existing.append(num)
            except (ValueError, IndexError):
                pass
    return sorted(existing)

def load_trial_status():
    if trial_status_file.exists():
        try:
            with open(trial_status_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
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
    """
    返回文件系统上标记为 running/interrupted 的 fs trial 列表（优先按编号升序）。
    pruned/completed/failed 会被跳过（不会当作要 resume 的对象）。
    """
    trial_status = load_trial_status()
    existing = get_existing_trial_numbers()
    nums = [int(trial_num) for trial_num, info in trial_status.items()
            if info["status"] in ["running", "interrupted"] and int(trial_num) in existing]
    nums.sort()
    return nums

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
# 将 HYPERPARAM_SPACE 转为 optuna.distributions dict（用于 create_trial）
# ===========================
def build_distributions_from_space():
    d = {}
    for k, v in HYPERPARAM_SPACE.items():
        if v["type"] == "float":
            d[k] = FloatDistribution(v["low"], v["high"], log=v.get("log", False))
        elif v["type"] == "categorical":
            d[k] = CategoricalDistribution(v["choices"])
        else:
            d[k] = CategoricalDistribution(v.get("choices", []))
    return d

# ===========================
# 读写 checkpoint / epoch helper（提前定义以便 clean_invalid_trials 调用）
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
# 从文件系统按顺序重建/替换 Optuna DB（确保 fs_trial -> optuna trial 映射）
# 使用 create_trial + add_trial 导入已完成试验（官方建议）
# 改动：已完成->COMPLETE；未完成只选择一个作为 RUNNING 导入，其余跳过导入，避免并行恢复冲突
# ===========================
def clean_invalid_trials():
    existing_trials = get_existing_trial_numbers()
    if not existing_trials:
        print("未发现任何有效试验，从0开始")
        return -1

    max_valid_trial = max(existing_trials)
    print(f"检测到有效试验最大编号: trial_{max_valid_trial}")

    timestamp = int(time.time())
    tmp_db = work_dir / f"study_rebuilt_{timestamp}.db"

    try:
        new_storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{tmp_db}",
            engine_kwargs={"connect_args": {"check_same_thread": False}}
        )
        new_study = optuna.create_study(
            study_name="yolo_optuna",
            direction="maximize",
            storage=new_storage,
            sampler=TPESampler(seed=seed, n_startup_trials=30),
            pruner=optuna.pruners.NopPruner(),
            load_if_exists=True
        )
    except Exception as e:
        print(f"创建重建用的新数据库失败: {e}")
        safe_unlink(tmp_db)
        return max_valid_trial

    distributions = build_distributions_from_space()

    imported = 0
    incomplete_candidates = []  # 保存未完成但有 params 的候选，后续只取一个设为 RUNNING
    skipped_incomplete = []

    for trial_num in existing_trials:
        trial_dir = work_dir / f"trial_{trial_num}"
        params_file = trial_dir / "params.yaml"
        params = {}
        if params_file.exists():
            try:
                with open(params_file, 'r') as f:
                    params = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"加载 trial_{trial_num} params 失败: {e}")
                params = {}

        # 计算 last_epoch（可能为0）
        last_epoch, _ = get_last_epoch(trial_dir)
        val = calculate_best_score(trial_dir)

        if not params:
            print(f"跳过导入 trial_{trial_num}（无 params）")
            continue

        if last_epoch >= EPOCHS_PER_TRIAL:
            # 该 fs trial 已完成 —— 导入为 COMPLETE（官方做法）
            try:
                frozen = ot_trial.create_trial(
                    params=params,
                    distributions=distributions,
                    value=float(val),
                    state=TrialState.COMPLETE,
                    user_attrs={"fs_trial": int(trial_num), "last_epoch": int(last_epoch)}
                )
                new_study.add_trial(frozen)
                imported += 1
            except Exception as e:
                print(f"导入 trial_{trial_num} 到新 DB 时出错: {e}")
        else:
            # 未完成：收集为候选，后面我们只选择其中一个变成 RUNNING 导入
            incomplete_candidates.append((trial_num, params, int(last_epoch), float(val)))

    # 选择一个未完成的 trial 作为 RUNNING 导入（保证一次仅恢复一个 RUNNING）
    if incomplete_candidates:
        # 按编号升序选择第一个（你可以改成其它策略，例如选最大的编号）
        chosen = sorted(incomplete_candidates, key=lambda x: x[0])[0]
        trial_num, params, last_epoch_chosen, val_chosen = chosen
        try:
            frozen = ot_trial.create_trial(
                params=params,
                distributions=distributions,
                value=None,  # 未完成没有 final value
                state=TrialState.RUNNING,
                user_attrs={"fs_trial": int(trial_num), "last_epoch": int(last_epoch_chosen)}
            )
            new_study.add_trial(frozen)
            imported += 1
            print(f"将未完成的 fs trial_{trial_num} 导入为 RUNNING（优先恢复此 trial）。")
        except Exception as e:
            print(f"将未完成 trial_{trial_num} 导入为 RUNNING 时出错: {e}")

        # 把其余未完成的 trial 记录下来（不导入 DB，后续 resume 会逐个处理）
        for other in incomplete_candidates:
            if other[0] == trial_num:
                continue
            skipped_incomplete.append(other[0])

    print(f"已保留 {imported} 个有效试验（包括最多 1 个 RUNNING）（≤ trial_{max_valid_trial}）")
    if skipped_incomplete:
        print(f"跳过导入 {len(skipped_incomplete)} 个未完成试验（编号示例: {skipped_incomplete[:5]})，它们将在 resume 时按需继续。")

    try:
        if hasattr(new_storage, "engine") and new_storage.engine is not None:
            new_storage.engine.dispose()
    except Exception:
        pass

    try:
        # 备份老 db 并替换
        if db_path.exists():
            safe_unlink(db_path)
        if tmp_db.exists():
            tmp_db.rename(db_path)
    except Exception as e:
        print(f"替换 study.db 时出错: {e}")

    # 同步 used_params（从文件系统读取）
    valid_params = []
    for trial_num in existing_trials:
        trial_dir = work_dir / f"trial_{trial_num}"
        params_file = trial_dir / "params.yaml"
        if params_file.exists():
            try:
                with open(params_file, 'r') as f:
                    params = yaml.safe_load(f)
                    if params:
                        valid_params.append(params)
            except Exception as e:
                print(f"警告：加载 trial_{trial_num} 的参数失败: {e}")
    save_used_params(valid_params)
    print(f"已更新有效参数记录，共 {len(valid_params)} 个有效参数组合")

    # 清理 trial_status 文件，仅保留文件系统上存在的
    if trial_status_file.exists():
        try:
            with open(trial_status_file, 'r') as f:
                try:
                    status = json.load(f)
                except json.JSONDecodeError:
                    status = {}
            cleaned_status = {k: v for k, v in status.items() if int(k) in existing_trials}
            with open(trial_status_file, 'w') as f:
                json.dump(cleaned_status, f, indent=2)
            print(f"已清理试验状态文件，保留 {len(cleaned_status)} 个有效状态记录")
        except Exception as e:
            print(f"清理 trial_status_file 时出错: {e}")

    return max_valid_trial

# ===========================
# 参数采样 - 两阶段策略（优先 Optuna suggest，失败后随机生成 fallback）
# objective 内部优先使用 params.yaml（resume），否则执行 trial.suggest_*
# ===========================
def sample_params_with_trial(trial):
    params = {}
    for k, v in HYPERPARAM_SPACE.items():
        if v["type"] == "float":
            params[k] = trial.suggest_float(k, v["low"], v["high"], log=v.get("log", False))
        elif v["type"] == "categorical":
            params[k] = trial.suggest_categorical(k, v["choices"])
        else:
            params[k] = trial.suggest_categorical(k, v.get("choices", []))
    return params

# ===========================
# 目标函数（支持 fs_trial_num）
# ===========================
def objective(trial, fs_trial_num=None):
    fs_num = fs_trial_num if fs_trial_num is not None else trial.number
    optuna_display = FS_TO_OPTUNA.get(fs_num, getattr(trial, "number", None))
    trial_dir = work_dir / f"trial_{fs_num}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    params_file = trial_dir / "params.yaml"

    # 优先使用已有 params.yaml（resume）
    if params_file.exists():
        try:
            with open(params_file, 'r') as f:
                params = yaml.safe_load(f) or {}
        except Exception:
            params = {}
    else:
        params = sample_params_with_trial(trial)
        try:
            with open(params_file, 'w') as f:
                yaml.dump(params, f)
        except Exception:
            pass

    last_completed_epoch, last_ckpt = get_last_epoch(trial_dir)
    start_epoch = last_completed_epoch

    if start_epoch >= EPOCHS_PER_TRIAL:
        best_score = calculate_best_score(trial_dir)
        print(f"[fs trial {fs_num} | optuna trial {optuna_display}] 已完成，最优加权得分: {best_score:.4f}")
        return best_score

    remaining_epochs = EPOCHS_PER_TRIAL - start_epoch

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
                train_dir_check = trial_dir / "train"
                if not train_dir_check.exists():
                    train_dir_check = trial_dir
                results_file = train_dir_check / "results.csv"
                if results_file.exists():
                    try:
                        df = pd.read_csv(results_file)
                        if not df.empty:
                            row = df.iloc[-1]
                            try:
                                precision = float(row.get("metrics/precision(B)", 0.0))
                                recall = float(row.get("metrics/recall(B)", 0.0))
                                mAP50 = float(row.get("metrics/mAP50(B)", 0.0))
                                mAP5095 = float(row.get("metrics/mAP50-95(B)", 0.0))
                                current_epoch = int(row.get("epoch", 0))
                            except Exception:
                                continue

                            score = (mAP50 * METRIC_WEIGHTS["metrics/mAP50(B)"] +
                                     mAP5095 * METRIC_WEIGHTS["metrics/mAP50-95(B)"] +
                                     precision * METRIC_WEIGHTS["metrics/precision(B)"] +
                                     recall * METRIC_WEIGHTS["metrics/recall(B)"])

                            print(f"[fs trial {fs_num} | optuna trial {optuna_display}] | Epoch {current_epoch}/{EPOCHS_PER_TRIAL} | "
                                  f"precision={precision:.3f}, recall={recall:.3f}, "
                                  f"mAP50={mAP50:.3f}, mAP50-95={mAP5095:.3f}")

                            update_trial_status(fs_num, current_epoch)

                            # 报告给 optuna（供剪枝决策）
                            try:
                                trial.report(score, current_epoch)
                                if trial.should_prune():
                                    print(f"[fs trial {fs_num} | optuna trial {optuna_display}] 在 Epoch {current_epoch} 被剪枝")
                                    update_trial_status(fs_num, current_epoch, "pruned")
                                    process.terminate()
                                    raise optuna.TrialPruned()
                            except Exception:
                                # 如果无法 report/should_prune（例如不是有效的 Trial 对象），忽略
                                pass
                    except Exception as e:
                        print(f"读取 results.csv 出错: {e}")
        process.wait()

        if process.returncode != 0:
            update_trial_status(fs_num, start_epoch, "failed")
            best_score = calculate_best_score(trial_dir)
            return best_score
    except KeyboardInterrupt:
        process.terminate()
        update_trial_status(fs_num, start_epoch, "interrupted")
        raise

    best_score = calculate_best_score(trial_dir)
    update_trial_status(fs_num, EPOCHS_PER_TRIAL, "completed")
    print(f"[fs trial {fs_num} | optuna trial {optuna_display}] 完成，最优加权得分: {best_score:.4f}")
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
# 主流程
# ===========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=TRIALS_TO_RUN)
    parser.add_argument("--epochs", type=int, default=EPOCHS_PER_TRIAL)
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-trial", type=int, nargs="*", help="要恢复的trial编号(可多个)")
    parser.add_argument("--clean", action="store_true", help="清除所有中断状态并重新开始")
    args = parser.parse_args()

    max_valid_trial = clean_invalid_trials()

    TRIALS_TO_RUN = args.trials
    EPOCHS_PER_TRIAL = args.epochs
    sampler = TPESampler(seed=seed, n_startup_trials=30)
    pruner = MedianPruner(n_startup_trials=30) if args.prune else optuna.pruners.NopPruner()

    if args.clean:
        if trial_status_file.exists():
            trial_status_file.unlink()
        valid_params = []
        for trial_num in get_existing_trial_numbers():
            trial_dir = work_dir / f"trial_{trial_num}"
            params_file = trial_dir / "params.yaml"
            if params_file.exists():
                try:
                    with open(params_file, 'r') as f:
                        params = yaml.safe_load(f)
                        if params:
                            valid_params.append(params)
                except Exception:
                    pass
        save_used_params(valid_params)
        print("已清除trial状态，保留历史有效参数记录")

    storage_url = f"sqlite:///{db_path}"
    try:
        study = optuna.create_study(
            study_name="yolo_optuna",
            direction="maximize",
            storage=storage_url,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )
    except Exception as e:
        print(f"加载/创建 study 失败: {e}，尝试删除 db 并重新创建")
        safe_unlink(db_path)
        study = optuna.create_study(
            study_name="yolo_optuna",
            direction="maximize",
            storage=storage_url,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )

    # 构造 fs->optuna 映射（读取现有 DB 中的 user_attrs）
    try:
        for t in study.trials:
            try:
                fa = t.user_attrs.get("fs_trial", None)
            except Exception:
                fa = None
            if fa is not None:
                FS_TO_OPTUNA[int(fa)] = t.number
    except Exception:
        pass

    # ---- resume specific trials（指定恢复） ----
    if args.resume_trial:
        existing_trials = get_existing_trial_numbers()
        for trial_num in args.resume_trial:
            if trial_num not in existing_trials:
                print(f"跳过不存在的試驗: trial_{trial_num}")
                continue

            # 尝试优先复用 DB 中已有的 optuna trial（且状态允许继续）
            trial_obj = None
            optuna_id = None
            if trial_num in FS_TO_OPTUNA:
                optuna_id = FS_TO_OPTUNA[trial_num]
                # 查找该 trial 在 study.trials 的快照，判断状态
                mapped = next((t for t in study.trials if t.number == optuna_id), None)
                if mapped is not None and mapped.state not in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL):
                    # 尝试构造可继续的 Trial 对象
                    try:
                        trial_obj = OptunaTrial(study, optuna_id)
                        print(f"复用已有的 optuna trial {optuna_id}（继续未完成试验） -> 对应文件夹 trial_{trial_num} ...")
                    except Exception as e:
                        trial_obj = None
                        print(f"尝试复用 optuna trial {optuna_id} 失败（将回退到 ask/enqueue）：{e}")

            # 若无法复用已有 trial，则尝试使用 params.yaml enqueue -> ask 流程（官方推荐）
            if trial_obj is None:
                params_file = work_dir / f"trial_{trial_num}" / "params.yaml"
                if params_file.exists():
                    try:
                        with open(params_file, "r") as f:
                            params = yaml.safe_load(f) or {}
                    except Exception:
                        params = {}
                    if params:
                        try:
                            study.enqueue_trial(params, user_attrs={"fs_trial": int(trial_num)})
                        except Exception as e:
                            print(f"enqueue_trial 失败: {e}")
                        trial_obj = study.ask()
                        try:
                            trial_obj.set_user_attr("fs_trial", int(trial_num))
                            FS_TO_OPTUNA[int(trial_num)] = trial_obj.number
                        except Exception:
                            pass
                        print(f"恢复并使用已存在参数: fs trial {trial_num} -> optuna trial {trial_obj.number}")
                    else:
                        trial_obj = study.ask()
                        try:
                            trial_obj.set_user_attr("fs_trial", int(trial_num))
                            FS_TO_OPTUNA[int(trial_num)] = trial_obj.number
                        except Exception:
                            pass
                        print(f"恢复 fs trial {trial_num}（params 文件为空/读取失败），将重新采样参数 -> optuna trial {trial_obj.number}")
                else:
                    trial_obj = study.ask()
                    try:
                        trial_obj.set_user_attr("fs_trial", int(trial_num))
                        FS_TO_OPTUNA[int(trial_num)] = trial_obj.number
                    except Exception:
                        pass
                    print(f"恢复 fs trial {trial_num}（无 params.yaml），optuna trial {trial_obj.number}（由 sampler 采样）")

            # 执行并写回
            try:
                val = objective(trial_obj, fs_trial_num=trial_num)
                try:
                    study.tell(trial_obj, float(val))
                except Exception as e:
                    print(f"tell 写回数据库失败 (resume_trial): {e}")
            except optuna.TrialPruned:
                trial_dir = work_dir / f"trial_{trial_num}"
                best = calculate_best_score(trial_dir)
                try:
                    study.tell(trial_obj, float(best), state=TrialState.PRUNED)
                except Exception as e:
                    print(f"剪枝状态写回失败: {e}")
            except Exception as e:
                print(f"恢复 trial {trial_num} 时出错: {e}")
                traceback.print_exc()

    # ---- resume 所有未完成的 trials ----
    if args.resume:
        incomplete = get_incomplete_trials()
        if incomplete:
            print(f"找到未完成的 trials: {incomplete}")
        # 顺序逐个 resume（每次只尝试继续一个 RUNNING trial，完成后继续下一个）
        for trial_num in incomplete:
            # 优先复用已有映射并且该 optuna trial 状态允许继续
            trial_obj = None
            optuna_id = None
            if trial_num in FS_TO_OPTUNA:
                optuna_id = FS_TO_OPTUNA[trial_num]
                mapped = next((t for t in study.trials if t.number == optuna_id), None)
                if mapped is not None and mapped.state not in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL):
                    try:
                        trial_obj = OptunaTrial(study, optuna_id)
                        print(f"复用已有的 optuna trial {optuna_id}（继续未完成试驗） -> 对应文件夹 trial_{trial_num} ...")
                    except Exception as e:
                        trial_obj = None
                        print(f"尝试复用 optuna trial {optuna_id} 失败（将回退到 ask/enqueue）：{e}")

            if trial_obj is None:
                params_file = work_dir / f"trial_{trial_num}" / "params.yaml"
                if params_file.exists():
                    try:
                        with open(params_file, "r") as f:
                            params = yaml.safe_load(f) or {}
                    except Exception:
                        params = {}
                    if params:
                        try:
                            study.enqueue_trial(params, user_attrs={"fs_trial": int(trial_num)})
                        except Exception as e:
                            print(f"enqueue_trial 失败: {e}")
                        trial_obj = study.ask()
                        try:
                            trial_obj.set_user_attr("fs_trial", int(trial_num))
                            FS_TO_OPTUNA[int(trial_num)] = trial_obj.number
                        except Exception:
                            pass
                        print(f"恢复并使用已存在参数: fs trial {trial_num} -> optuna trial {trial_obj.number}")
                    else:
                        trial_obj = study.ask()
                        trial_obj.set_user_attr("fs_trial", int(trial_num))
                        FS_TO_OPTUNA[int(trial_num)] = trial_obj.number
                        print(f"恢复 fs trial {trial_num}，optuna trial {trial_obj.number}（将重新采样参数）")
                else:
                    trial_obj = study.ask()
                    trial_obj.set_user_attr("fs_trial", int(trial_num))
                    FS_TO_OPTUNA[int(trial_num)] = trial_obj.number
                    print(f"恢复 fs trial {trial_num}（无 params.yaml），optuna trial {trial_obj.number}（由 sampler 采样）")

            try:
                val = objective(trial_obj, fs_trial_num=trial_num)
                try:
                    study.tell(trial_obj, float(val))
                except Exception as e:
                    print(f"tell 写回数据库失败 (resume): {e}")
            except optuna.TrialPruned:
                best = calculate_best_score(work_dir / f"trial_{trial_num}")
                try:
                    study.tell(trial_obj, float(best), state=TrialState.PRUNED)
                except Exception as e:
                    print(f"剪枝写回失败: {e}")
            except Exception as e:
                print(f"恢复 trial {trial_num} 时出错: {e}")
                traceback.print_exc()

    # ---- 创建新的 trials（补足到指定数量） ----
    existing_trials = get_existing_trial_numbers()
    existing_count = len(existing_trials)
    next_trials_needed = max(0, TRIALS_TO_RUN - existing_count)
    if next_trials_needed > 0:
        print(f"需要 {next_trials_needed} 个新 trials ...")
        start_trial_number = (max_valid_trial + 1) if max_valid_trial >= 0 else 0
        for i in range(next_trials_needed):
            fs_trial_number = start_trial_number + i
            if fs_trial_number in get_existing_trial_numbers():
                print(f"trial_{fs_trial_number} 已存在，跳过...")
                continue

            params_file = work_dir / f"trial_{fs_trial_number}" / "params.yaml"
            if params_file.exists():
                try:
                    with open(params_file, "r") as f:
                        params = yaml.safe_load(f) or {}
                except Exception:
                    params = {}
                if params:
                    try:
                        study.enqueue_trial(params, user_attrs={"fs_trial": int(fs_trial_number)})
                    except Exception as e:
                        print(f"enqueue_trial 失败 (new trial): {e}")

            trial_obj = study.ask()
            try:
                trial_obj.set_user_attr("fs_trial", int(fs_trial_number))
                FS_TO_OPTUNA[int(fs_trial_number)] = trial_obj.number
            except Exception:
                pass
            print(f"开始新 fs trial {fs_trial_number} (optuna trial {trial_obj.number}) ...")
            try:
                val = objective(trial_obj, fs_trial_num=fs_trial_number)
                try:
                    study.tell(trial_obj, float(val))
                except Exception as e:
                    print(f"tell 写回数据库失败 (new trial): {e}")
            except optuna.TrialPruned:
                best = calculate_best_score(work_dir / f"trial_{fs_trial_number}")
                try:
                    study.tell(trial_obj, float(best), state=TrialState.PRUNED)
                except Exception as e:
                    print(f"剪枝写回失败: {e}")
            except Exception as e:
                print(f"运行 trial {fs_trial_number} 时出错: {e}")
                traceback.print_exc()

    print("=== 所有 trials 完成（或已调度） ===")

    # 重新 load study（确保最新结果）
    study = optuna.create_study(
        study_name="yolo_optuna",
        direction="maximize",
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )

    try:
        if study.best_trial:
            print(f"最佳 optuna trial: {study.best_trial.number}, 分数={study.best_trial.value}")
            print(f"最佳参数: {study.best_trial.params}")
        else:
            print("没有找到最佳试验")
    except Exception as e:
        print(f"获取最佳 trial 失败: {e}")
