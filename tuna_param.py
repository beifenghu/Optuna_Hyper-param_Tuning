# tune_param_fixed.py
# 修复要点：
# - 在写入 storage 前检查 trial 在 storage 中的 state（若已结束则跳过写入 storage）
# - 在创建/复用 trial 并开始训练前，立即把对应 trial_{N}/params.yaml 的参数写入 trial（优先 storage，再回退到 trial._params）
# - 在对 trial 调用 study.tell 前，使用 safe_tell() 检查 storage 状态并避免对已 COMPLETE 的 trial 进行 tell（防止 "Cannot tell a COMPLETE trial."）

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
data_yaml = ".yaml"
base_model = "yolo11s.pt"
trials = 50
epochs_per_trial = 50
work_dir = Path("optuna_temp")
work_dir.mkdir(exist_ok=True)
trial_status_file = work_dir / "trial_status.json"
seed = 42
db_path = work_dir / "study.db"
USED_PARAMS_FILE = work_dir / "used_params.json"

# global study (will be set in main)
study = None

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

FS_TO_OPTUNA = {}

# ===========================
# 工具函数（多数沿用你原来的实现）
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

def _make_distribution_for_param(k):
    info = HYPERPARAM_SPACE.get(k)
    if info is None:
        return None
    if info["type"] == "float":
        return FloatDistribution(low=info["low"], high=info["high"], log=info.get("log", False))
    elif info["type"] == "categorical":
        return CategoricalDistribution(info["choices"])
    else:
        return None

# ===========================
# 新增辅助：读取 storage 中某个 trial 的 state（若可查）
# ===========================
def _get_storage_trial_state(study_obj, trial_number):
    """
    返回 storage 中 trial_number 对应的 state（TrialState 枚举），如果无法查到返回 None。
    """
    try:
        storage_trials = study_obj._storage.get_all_trials(study_obj._study_id)
        for st in storage_trials:
            try:
                if getattr(st, "number", None) == trial_number:
                    return getattr(st, "state", None)
            except Exception:
                continue
    except Exception:
        return None
    return None

# ===========================
# 安全 tell：在写回 study 前检查 storage 中 trial 状态
# ===========================
def safe_tell(study_obj, trial_obj, value, state=None):
    """
    在调用 study.tell 前检查 storage 中该 trial 是否已经 COMPLETE/PRUNED/FAIL，
    如果已结束则跳过 tell（避免 'Cannot tell a COMPLETE trial.' 错误）。
    """
    trial_number = getattr(trial_obj, "number", None)
    try:
        storage_state = _get_storage_trial_state(study_obj, trial_number)
        if storage_state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL):
            # trial 在 storage 中已结束：跳过 tell
            print(f"safe_tell: storage 中 trial#{trial_number} 已结束（{storage_state}），跳过 tell。")
            return
        # 否则正常写回
        if state is None:
            study_obj.tell(trial_obj, value)
        else:
            study_obj.tell(trial_obj, value, state=state)
    except Exception as e:
        # 若出现 "Cannot tell a COMPLETE trial." 之类的异常，打印并继续（不抛出）
        print(f"safe_tell: 写回 trial#{trial_number} 时出错: {e}")

# ===========================
# 将 params 强制写入 trial（尽量写入 storage，否则回退到 trial._params）
# ===========================
def apply_params_to_trial_storage(trial_obj, params):
    """
    优先尝试写入 storage（使用分布的内部表示），写入前检查 storage 中该 trial 是否已结束，
    已结束就直接把 params 保存在 trial._params（回退），避免 storage 抛错噪声。
    """
    global study
    # 先保证 trial_obj._params 更新（内存优先）
    try:
        if hasattr(trial_obj, "_params"):
            trial_obj._params = trial_obj._params or {}
            trial_obj._params.update(params)
        else:
            setattr(trial_obj, "_params", params.copy())
    except Exception:
        pass

    if study is None:
        return

    trial_number = getattr(trial_obj, "number", None)
    # 获取 storage 中对应 trial 的 state
    try:
        storage_state = _get_storage_trial_state(study, trial_number)
    except Exception:
        storage_state = None

    if storage_state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL):
        # storage 中该 trial 已结束：不再试图 set_trial_param，直接返回（不过 _params 已写）
        print(f"apply_params_to_trial_storage: storage 中 trial#{trial_number} 已结束，跳过 set_trial_param（使用 params.yaml 作为真相）")
        return

    trial_id_for_storage = getattr(trial_obj, "_trial_id", None)
    if trial_id_for_storage is None:
        trial_id_for_storage = trial_number

    # 逐参数写入 storage（转换为内部表示）
    for k, v in params.items():
        dist = _make_distribution_for_param(k)
        if dist is None:
            # 无法构建 distribution：保留在 _params（已回退）
            continue
        try:
            internal_val = dist.to_internal_repr(v)
        except Exception as e:
            print(f"apply_params_to_trial_storage: 参数 {k}={v} 转内部表示失败: {e}")
            continue
        try:
            study._storage.set_trial_param(trial_id_for_storage, k, internal_val, dist)
        except Exception as e:
            # 若 storage 在写入时变为已结束，静默回退（打印一次信息）
            msg = str(e)
            if "has already finished" in msg or "can not be updated" in msg or "finished" in msg:
                print(f"apply_params_to_trial_storage: 无法通过 storage.set_trial_param 写入 trial#{trial_number} (可能已结束)：{e}。已回退到 trial._params。")
                return
            else:
                print(f"apply_params_to_trial_storage: 无法写入参数 {k} 到 storage: {e}")
                # 回退到 _params（已经写过，但再次确保）
                try:
                    if hasattr(trial_obj, "_params"):
                        trial_obj._params = trial_obj._params or {}
                        trial_obj._params[k] = v
                    else:
                        setattr(trial_obj, "_params", {k: v})
                except Exception:
                    pass

# ===========================
# smart sync：把文件系统 params 写回 DB（仅对未完成 trial 尝试写入 storage）
# ===========================
def smart_sync_params_from_fs_to_db(study_local):
    existing_trials = study_local.trials
    written = 0
    skipped_finished = 0
    for t in existing_trials:
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

        state = getattr(t, "state", None)
        if state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL):
            # 结束的 trial：只回写到 t._params，跳过 storage 写入
            try:
                if hasattr(t, "_params"):
                    t._params = t._params or {}
                    t._params.update(params)
                else:
                    setattr(t, "_params", params.copy())
            except Exception:
                pass
            skipped_finished += 1
            continue

        try:
            apply_params_to_trial_storage(t, params)
            written += 1
        except Exception as e:
            print(f"smart_sync: 写入 trial {t.number} 失败: {e}")

    print(f"smart_sync_params_from_fs_to_db: 写入 storage {written} 个 trial，跳过 (finished) {skipped_finished} 个")

# ===========================
# clean_invalid_trials（保留你现有的逻辑，写入时使用内部表示）
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
                    internal_val = dist.to_internal_repr(v)
                    new_study._storage.set_trial_param(trial_id_for_storage, k, internal_val, dist)
                except Exception as e:
                    try:
                        if hasattr(t, "_params"):
                            t._params = t._params or {}
                            t._params[k] = v
                        else:
                            setattr(t, "_params", {k: v})
                    except Exception:
                        pass
                    print(f"重建时：无法 storage.set_trial_param 写入 trial fs={trial_num} 的参数 {k}，原因: {e}")

            try:
                t.set_user_attr("fs_trial", int(trial_num))
            except Exception:
                pass

            new_study.tell(t, float(val))
            imported += 1
        except Exception as e:
            print(f"导入 trial_{trial_num} 到新 DB 时出错: {e}")

    print(f"已保留 {imported} 个有效试验（≤ trial_{max_valid_trial}）")

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
# sample_params / objective / get_last_epoch / signal handler 等（保持逻辑，只在关键点调用 apply_params_to_trial_storage / safe_tell）
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

def objective(trial, fs_trial_num=None):
    global study
    fs_num = fs_trial_num if fs_trial_num is not None else trial.number
    optuna_display = FS_TO_OPTUNA.get(fs_num, getattr(trial, "number", None))
    trial_dir = work_dir / f"trial_{fs_num}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    params_file = trial_dir / "params.yaml"

    if params_file.exists():
        try:
            with open(params_file, "r") as f:
                params = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[fs trial {fs_num}] 读取 params.yaml 失败: {e}")
            params = None
    else:
        params = None

    if params is None or not params:
        params = sample_params(trial)
        try:
            with open(params_file, "w") as f:
                yaml.dump(params, f)
        except Exception as e:
            print(f"[fs trial {fs_num}] 写 params.yaml 失败: {e}")
        # 把新采样的 params 立即写入 trial（storage 或 _params）
        try:
            apply_params_to_trial_storage(trial, params)
        except Exception as e:
            print(f"[fs trial {fs_num}] 将新采样参数写入 storage 失败: {e}")
    else:
        # file 存在时，强制把 params.yaml 内容写回 trial（确保 DB 与 filesystem 保持一致）
        try:
            apply_params_to_trial_storage(trial, params)
        except Exception as e:
            print(f"[fs trial {fs_num}] 强制将 params.yaml 写回 trial storage 失败: {e}")

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

                            try:
                                trial.report(score, current_epoch)
                                if trial.should_prune():
                                    print(f"[fs trial {fs_num} | optuna trial {optuna_display}] 在 Epoch {current_epoch} 被剪枝")
                                    update_trial_status(fs_num, current_epoch, "pruned")
                                    process.terminate()
                                    raise optuna.TrialPruned()
                            except Exception:
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
# 主流程（关键处调用 apply_params_to_trial_storage & safe_tell）
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

    # 把 filesystem 的 params 强制同步到 DB（智能版）
    try:
        smart_sync_params_from_fs_to_db(study)
    except Exception as e:
        print(f"smart_sync 出错: {e}")

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

            # 在开始 objective 前，确保将 filesystem 的 params 写入该 trial（避免 race）
            params_path = work_dir / f"trial_{trial_num}" / "params.yaml"
            if params_path.exists():
                try:
                    with open(params_path, "r") as f:
                        params = yaml.safe_load(f) or {}
                    if params:
                        try:
                            apply_params_to_trial_storage(trial_obj, params)
                        except Exception:
                            pass
                except Exception:
                    pass

            try:
                val = objective(trial_obj, fs_trial_num=trial_num)
                try:
                    safe_tell(study, trial_obj, float(val))
                except Exception as e:
                    print(f"tell 写回数据库失败 (resume_trial): {e}")
            except optuna.TrialPruned:
                trial_dir = work_dir / f"trial_{trial_num}"
                best = calculate_best_score(trial_dir)
                try:
                    safe_tell(study, trial_obj, float(best), state=TrialState.PRUNED)
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
                except Exception:
                    pass
                print(f"恢复 optuna trial {trial_obj.number} -> 对应文件夹 trial_{trial_num} ...")

            # 在开始 objective 前，确保将 filesystem 的 params 写入该 trial（避免 race）
            params_path = work_dir / f"trial_{trial_num}" / "params.yaml"
            if params_path.exists():
                try:
                    with open(params_path, "r") as f:
                        params = yaml.safe_load(f) or {}
                    if params:
                        try:
                            apply_params_to_trial_storage(trial_obj, params)
                        except Exception:
                            pass
                except Exception:
                    pass

            try:
                val = objective(trial_obj, fs_trial_num=trial_num)
                try:
                    safe_tell(study, trial_obj, float(val))
                except Exception as e:
                    print(f"tell 写回数据库失败 (resume): {e}")
            except optuna.TrialPruned:
                best = calculate_best_score(work_dir / f"trial_{trial_num}")
                try:
                    safe_tell(study, trial_obj, float(best), state=TrialState.PRUNED)
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

            # **关键改动**：在 objective 前，优先把 filesystem 的 params 写入该新 trial（避免新 trial 创建后被其他流程标记 COMPLETE 等 race）
            params_path = work_dir / f"trial_{fs_trial_number}" / "params.yaml"
            if params_path.exists():
                try:
                    with open(params_path, "r") as f:
                        params = yaml.safe_load(f) or {}
                    if params:
                        try:
                            apply_params_to_trial_storage(trial_obj, params)
                        except Exception:
                            pass
                except Exception:
                    pass

            try:
                val = objective(trial_obj, fs_trial_num=fs_trial_number)
                trial_dir = work_dir / f"trial_{fs_trial_number}"
                params_file = trial_dir / "params.yaml"
                if params_file.exists():
                    try:
                        params = yaml.safe_load(open(params_file, "r"))
                        try:
                            # 保证 trial._params 也保留一致数据
                            if hasattr(trial_obj, "_params"):
                                trial_obj._params = trial_obj._params or {}
                                trial_obj._params.update(params)
                            else:
                                trial_obj._params = params
                        except Exception:
                            pass
                    except Exception:
                        pass
                try:
                    safe_tell(study, trial_obj, float(val))
                except Exception as e:
                    print(f"tell 写回数据库失败 (new trial): {e}")
            except optuna.TrialPruned:
                best = calculate_best_score(work_dir / f"trial_{fs_trial_number}")
                try:
                    safe_tell(study, trial_obj, float(best), state=TrialState.PRUNED)
                except Exception as e:
                    print(f"剪枝写回失败: {e}")
            except Exception as e:
                print(f"运行 trial {fs_trial_number} 时出错: {e}")

    print("=== 所有 trials 完成（或已调度） ===")

    # 重新加载 study 并输出最佳 trial（保留原逻辑）
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
