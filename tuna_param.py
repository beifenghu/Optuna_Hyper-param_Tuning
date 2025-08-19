# fix_tune_param.py
# 修复版：确保写入 Optuna storage 时使用 distribution.to_internal_repr(...)
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

# ===========================
# 配置路径和训练参数
# ===========================
data_yaml = "chengdu.yaml"
base_model = "yolo11s.pt"
trials = 50
epochs_per_trial = 50
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

# 全局映射 fs_trial -> optuna trial number（由 DB 中的 user_attr 或者创建时填充）
FS_TO_OPTUNA = {}

# ===========================
# 工具函数
# ===========================
def safe_unlink(path, max_attempts=5, delay=1):
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
# 新增: 构造分布并同步 params 到 Optuna DB
# ===========================
def _make_distribution_for_param(k):
    """根据 HYPERPARAM_SPACE 为 k 构造 optuna distribution"""
    info = HYPERPARAM_SPACE.get(k)
    if info is None:
        return None
    if info["type"] == "float":
        return FloatDistribution(low=info["low"], high=info["high"], log=info.get("log", False))
    elif info["type"] == "categorical":
        return CategoricalDistribution(info["choices"])
    else:
        return None

def sync_params_from_fs_to_db(study):
    """
    把 work_dir/trial_{fs}/params.yaml 的参数写回到 optuna DB 中对应的 trial。
    匹配逻辑：优先使用 trial.user_attrs["fs_trial"]，否则尝试以 trial.number 对应的文件名（trial_{number}）。
    """
    written = 0
    for t in study.trials:
        try:
            fs_trial = t.user_attrs.get("fs_trial", None)
        except Exception:
            fs_trial = None

        if fs_trial is None:
            candidate_fs = t.number
        else:
            try:
                candidate_fs = int(fs_trial)
            except Exception:
                candidate_fs = t.number

        params_path = work_dir / f"trial_{candidate_fs}" / "params.yaml"
        if not params_path.exists():
            continue

        try:
            with open(params_path, "r") as f:
                params = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"读取 {params_path} 失败: {e}")
            continue

        # 如果 DB 中已存在 params，可选择跳过以免覆盖。这里保守：只有在 DB params 为空时写入。
        try:
            if t.params and len(t.params) > 0:
                continue
        except Exception:
            pass

        for k, v in params.items():
            dist = _make_distribution_for_param(k)
            if dist is None:
                try:
                    if hasattr(t, "_params"):
                        t._params = t._params or {}
                        t._params[k] = v
                    else:
                        setattr(t, "_params", {k: v})
                except Exception:
                    pass
                continue
            try:
                trial_id_for_storage = getattr(t, "_trial_id", None)
                if trial_id_for_storage is None:
                    trial_id_for_storage = t.number
                # 转换外部值 -> 内部表示
                try:
                    internal = dist.to_internal_repr(v)
                except Exception as e:
                    print(f"参数 {k}={v} 转换失败: {e}")
                    # 退回到直接写 _params
                    try:
                        if hasattr(t, "_params"):
                            t._params = t._params or {}
                            t._params[k] = v
                        else:
                            setattr(t, "_params", {k: v})
                    except Exception:
                        pass
                    continue

                # 写入 storage（内部表示）
                try:
                    study._storage.set_trial_param(trial_id_for_storage, k, internal, dist)
                except Exception as e:
                    # 退回到 _params 以兼容
                    try:
                        if hasattr(t, "_params"):
                            t._params = t._params or {}
                            t._params[k] = v
                        else:
                            setattr(t, "_params", {k: v})
                    except Exception:
                        pass
                    print(f"警告：无法通过 storage.set_trial_param 写入 trial {t.number} 的参数 {k}，原因: {e}")
            except Exception as e:
                try:
                    if hasattr(t, "_params"):
                        t._params = t._params or {}
                        t._params[k] = v
                    else:
                        setattr(t, "_params", {k: v})
                except Exception:
                    pass

        try:
            if hasattr(t, "_params"):
                t._params = t._params or {}
                t._params.update(params)
            else:
                setattr(t, "_params", params)
        except Exception:
            pass

        written += 1

    print(f"sync_params_from_fs_to_db: 完成, 写入或尝试写入 {written} 个 trial 的参数（若存在）")

# ===========================
# 从文件系统按顺序重建/替换 Optuna DB（确保 fs_trial -> optuna trial 映射 & 写入 params）
# ===========================
def clean_invalid_trials():
    """
    从文件系统（work_dir/trial_*/params.yaml + results.csv）重建 study.db，
    按 trial_0, trial_1 ... 的顺序把每个 trial 导入 DB，并写入：
      - user_attr "fs_trial"
      - params（基于 params.yaml + HYPERPARAM_SPACE 的 Distribution）
    返回 max_valid_trial（文件系统最大编号）。
    """
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

    imported = 0
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

        val = calculate_best_score(trial_dir)
        try:
            t = new_study.ask()

            # 写 params 到新 DB —— **重要：先转换为内部表示**
            for k, v in params.items():
                dist = _make_distribution_for_param(k)
                if dist is None:
                    # 跳过 distribution 无法构建的参数（写入 _params）
                    continue
                try:
                    trial_id_for_storage = getattr(t, "_trial_id", None)
                    if trial_id_for_storage is None:
                        trial_id_for_storage = t.number
                    # 关键改动：使用 dist.to_internal_repr 转换外部值为内部表示
                    try:
                        internal_val = dist.to_internal_repr(v)
                    except Exception as e:
                        print(f"参数 {k}={v} 在内部表示转换失败: {e}")
                        # 回退：保存到 _params
                        try:
                            if hasattr(t, "_params"):
                                t._params = t._params or {}
                                t._params[k] = v
                            else:
                                setattr(t, "_params", {k: v})
                        except Exception:
                            pass
                        continue

                    # 写入 storage（内部表示）
                    try:
                        new_study._storage.set_trial_param(trial_id_for_storage, k, internal_val, dist)
                    except Exception as e:
                        # 回退：保存到 _params
                        try:
                            if hasattr(t, "_params"):
                                t._params = t._params or {}
                                t._params[k] = v
                            else:
                                setattr(t, "_params", {k: v})
                        except Exception:
                            pass
                        print(f"重建时：无法 storage.set_trial_param 写入 trial fs={trial_num} 的参数 {k}，原因: {e}")
                except Exception as e:
                    print(f"重建时: 设置参数 {k} 失败: {e}")

            # 记录 fs_trial 映射
            try:
                t.set_user_attr("fs_trial", int(trial_num))
            except Exception:
                pass

            new_study.tell(t, float(val))
            imported += 1
        except Exception as e:
            print(f"导入 trial_{trial_num} 到新 DB 时出错: {e}")

    print(f"已保留 {imported} 个有效试验（≤ trial_{max_valid_trial}）")

    # 关闭并替换数据库
    try:
        if hasattr(new_storage, "engine") and new_storage.engine is not None:
            new_storage.engine.dispose()
    except Exception:
        pass

    try:
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
# 参数采样 - 两阶段策略（优先 Optuna suggest，失败后随机生成）
# ===========================
def sample_params(trial):
    used_params = load_used_params()
    for params_path in work_dir.glob("trial_*/params.yaml"):
        try:
            with open(params_path, "r") as f:
                fs_params = yaml.safe_load(f)
                if fs_params and not is_param_used(fs_params, used_params):
                    used_params.append(fs_params)
        except Exception as e:
            print(f"加载 {params_path} 时出错: {e}")
    save_used_params(used_params)

    max_optuna_attempts = 120
    attempts = 0
    while attempts < max_optuna_attempts:
        attempts += 1
        params = {}
        for k, v in HYPERPARAM_SPACE.items():
            if v["type"] == "float":
                params[k] = trial.suggest_float(k, v["low"], v["high"], log=v.get("log", False))
            elif v["type"] == "categorical":
                params[k] = trial.suggest_categorical(k, v["choices"])
        if not is_param_used(params, used_params):
            used_params.append(params)
            save_used_params(used_params)
            return params
    print(f"Optuna sampler 在 {max_optuna_attempts} 次尝试中无法找到新参数，转入随机采样 fallback。")

    max_random_attempts = 200
    rand_attempts = 0
    while rand_attempts < max_random_attempts:
        rand_attempts += 1
        params = {}
        for k, v in HYPERPARAM_SPACE.items():
            if v["type"] == "float":
                if v.get("log", False):
                    low_log = math.log10(v["low"])
                    high_log = math.log10(v["high"])
                    val = 10 ** random.uniform(low_log, high_log)
                    params[k] = float(val)
                else:
                    params[k] = float(random.uniform(v["low"], v["high"]))
            elif v["type"] == "categorical":
                params[k] = random.choice(v["choices"])
        if not is_param_used(params, used_params):
            used_params.append(params)
            save_used_params(used_params)
            print(f"随机采样成功（尝试 {rand_attempts} 次），采用新参数。")
            return params

    print(f"警告：随机采样也在 {max_random_attempts} 次尝试后未找到新参数，可能返回重复参数")
    used_params.append(params)
    save_used_params(used_params)
    return params

# ===========================
# 目标函数（支持 fs_trial_num）
# ===========================
def objective(trial, fs_trial_num=None):
    fs_num = fs_trial_num if fs_trial_num is not None else trial.number
    # 显示的 optuna id：优先用映射（如果存在），否则使用 trial.number
    optuna_display = FS_TO_OPTUNA.get(fs_num, getattr(trial, "number", None))
    trial_num = getattr(trial, "number", None)
    trial_dir = work_dir / f"trial_{fs_num}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    params_file = trial_dir / "params.yaml"

    if params_file.exists():
        with open(params_file, 'r') as f:
            params = yaml.safe_load(f)
    else:
        params = sample_params(trial)
        with open(params_file, 'w') as f:
            yaml.dump(params, f)

    last_completed_epoch, last_ckpt = get_last_epoch(trial_dir)
    start_epoch = last_completed_epoch

    if start_epoch >= epochs_per_trial:
        best_score = calculate_best_score(trial_dir)
        print(f"[fs trial {fs_num} | optuna trial {optuna_display}] 已完成，最优加权得分: {best_score:.4f}")
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

                            print(f"[fs trial {fs_num} | optuna trial {optuna_display}] | Epoch {current_epoch}/{epochs_per_trial} | "
                                  f"precision={precision:.3f}, recall={recall:.3f}, "
                                  f"mAP50={mAP50:.3f}, mAP50-95={mAP5095:.3f}")

                            update_trial_status(fs_num, current_epoch)

                            # 报告给 optuna（尽量保持 prune 行为）
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
    update_trial_status(fs_num, epochs_per_trial, "completed")
    print(f"[fs trial {fs_num} | optuna trial {optuna_display}] 完成，最优加权得分: {best_score:.4f}")
    return best_score

# ===========================
# 读写 checkpoint / epoch helper
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
    parser.add_argument("--trials", type=int, default=trials)
    parser.add_argument("--epochs", type=int, default=epochs_per_trial)
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-trial", type=int, nargs="*", help="要恢复的trial编号(可多个)")
    parser.add_argument("--clean", action="store_true", help="清除所有中断状态并重新开始")
    args = parser.parse_args()

    max_valid_trial = clean_invalid_trials()

    trials = args.trials
    epochs_per_trial = args.epochs
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

    # 新增：尝试把 filesystem 上的 params.yaml 写回到 optuna DB（避免 DB 中 params 为空）
    try:
        sync_params_from_fs_to_db(study)
    except Exception as e:
        print(f"同步 params 到 DB 失败: {e}")

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

    # ---- resume specific trials 指定恢复 ----
    if args.resume_trial:
        existing_trials = get_existing_trial_numbers()
        for trial_num in args.resume_trial:
            if trial_num not in existing_trials:
                print(f"跳过不存在的試驗: trial_{trial_num}")
                continue

            # 优先复用 DB 中已有 mapping 的 optuna trial（如果存在）
            reused_trial_obj = None
            optuna_id = None
            if trial_num in FS_TO_OPTUNA:
                optuna_id = FS_TO_OPTUNA[trial_num]
                try:
                    reused_trial_obj = OptunaTrial(study, optuna_id)
                except Exception:
                    reused_trial_obj = None

            if reused_trial_obj is not None:
                trial_obj = reused_trial_obj
                print(f"复用已存在的 optuna trial {optuna_id} -> 对应文件夹 trial_{trial_num} ...")
            else:
                trial_obj = study.ask()
                try:
                    trial_obj.set_user_attr("fs_trial", int(trial_num))
                    FS_TO_OPTUNA[int(trial_num)] = trial_obj.number
                    optuna_id = trial_obj.number
                except Exception:
                    pass
                print(f"恢复 optuna trial {trial_obj.number} -> 对应文件夹 trial_{trial_num} ...")

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

    # ---- resume 所有未完成的 trials ----
    if args.resume:
        incomplete = get_incomplete_trials()
        if incomplete:
            print(f"找到未完成的 trials: {incomplete}")
        for trial_num in incomplete:
            reused_trial_obj = None
            optuna_id = None
            if trial_num in FS_TO_OPTUNA:
                optuna_id = FS_TO_OPTUNA[trial_num]
                try:
                    reused_trial_obj = OptunaTrial(study, optuna_id)
                except Exception:
                    reused_trial_obj = None

            if reused_trial_obj is not None:
                trial_obj = reused_trial_obj
                print(f"复用已存在的 optuna trial {optuna_id} -> 对应文件夹 trial_{trial_num} ...")
            else:
                trial_obj = study.ask()
                try:
                    trial_obj.set_user_attr("fs_trial", int(trial_num))
                    FS_TO_OPTUNA[int(trial_num)] = trial_obj.number
                    optuna_id = trial_obj.number
                except Exception:
                    pass
                print(f"恢复 optuna trial {trial_obj.number} -> 对应文件夹 trial_{trial_num} ...")

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

    # ---- 创建新的 trials（补足到指定数量） ----
    existing_trials = get_existing_trial_numbers()
    existing_count = len(existing_trials)
    next_trials_needed = max(0, trials - existing_count)
    if next_trials_needed > 0:
        print(f"需要 {next_trials_needed} 个新 trials ...")
        start_trial_number = (max_valid_trial + 1) if max_valid_trial >= 0 else 0
        for i in range(next_trials_needed):
            fs_trial_number = start_trial_number + i
            if fs_trial_number in get_existing_trial_numbers():
                print(f"trial_{fs_trial_number} 已存在，跳过...")
                continue
            trial_obj = study.ask()
            try:
                trial_obj.set_user_attr("fs_trial", int(fs_trial_number))
                FS_TO_OPTUNA[int(fs_trial_number)] = trial_obj.number
            except Exception:
                pass
            print(f"开始新 fs trial {fs_trial_number} (optuna trial {trial_obj.number}) ...")
            try:
                val = objective(trial_obj, fs_trial_num=fs_trial_number)
                trial_dir = work_dir / f"trial_{fs_trial_number}"
                params_file = trial_dir / "params.yaml"
                if params_file.exists():
                    try:
                        params = yaml.safe_load(open(params_file, "r"))
                        try:
                            trial_obj._params = params
                        except Exception:
                            pass
                    except Exception:
                        pass
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

    print("=== 所有 trials 完成（或已调度） ===")

    # 重新 load study
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
