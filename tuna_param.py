#!/usr/bin/env python3
# coding: utf-8
"""
Optuna 恢复逻辑脚本（改进）
- 重点修复：训练完成后把结果可靠地写回到 DB 中映射到该 fs_trial 的 optuna trial（及时把 RUNNING -> COMPLETE）
- 所有写回均通过 finalize_fs_trial(...) 统一处理，包含多次重试与 storage 回退写入与写回后核验
- 其余逻辑（clean/resume/new trial 等）尽量不变
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
# 配置（按需修改）
# ===========================
data_yaml = "chengdu.yaml"
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

# 全局映射 fs_trial -> optuna trial number（内存缓存）
FS_TO_OPTUNA = {}

# ===========================
# 工具函数
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
                print(f"警告：无法删除文件 {path}, 可能被其他进程占用")
                return False
        except Exception as e:
            print(f"删除文件 {path} 时出错: {e}")
            return False
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
            except Exception:
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

def get_incomplete_trials_from_status():
    trial_status = load_trial_status()
    existing = get_existing_trial_numbers()
    nums = [int(trial_num) for trial_num, info in trial_status.items() if info["status"] in ["running", "interrupted"] and int(trial_num) in existing]
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
# 刷新内存映射（从 DB 中读取 user_attr["fs_trial"]）
# ===========================
def refresh_fs_to_optuna(study):
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

# ===========================
# safe storage 写入：尽最大努力把 value/state 持久化到 storage
# - 会尝试使用 storage.get_all_trials(study_id) 来定位内部 id
# - 然后调用 set_trial_value / set_trial_state 等 API（若存在）
# - 写入后不忘重载 study 并核验写入结果
# ===========================
def safe_storage_write_and_verify(study, fs_num, mapped_number, value, state, tries=3, sleep_between=0.5):
    """
    返回 True 表示确认 DB 中对应 trial(value/state) 已被写入并核验。
    """
    try:
        storage = study._storage
    except Exception as e:
        print(f"无法获取 study._storage：{e}")
        return False

    # 尝试定位内部 id（优先通过 storage.get_all_trials）
    tid = None
    trial_frozen = None
    try:
        if hasattr(storage, "get_all_trials") and hasattr(study, "_study_id"):
            try:
                all_trials = storage.get_all_trials(study._study_id)
                for ft in all_trials:
                    # Some storage FrozenTrial may have 'number' attr
                    if getattr(ft, "number", None) == mapped_number:
                        trial_frozen = ft
                        tid = getattr(ft, "_trial_id", None) or getattr(ft, "trial_id", None) or getattr(ft, "id", None)
                        break
            except Exception:
                # some storages don't support get_all_trials with study id param
                try:
                    all_trials = storage.get_all_trials()
                    for ft in all_trials:
                        if getattr(ft, "number", None) == mapped_number:
                            trial_frozen = ft
                            tid = getattr(ft, "_trial_id", None) or getattr(ft, "trial_id", None) or getattr(ft, "id", None)
                            break
                except Exception:
                    pass
        # fallback: if we were given a frozen trial object earlier, try to take id from it
        if tid is None and trial_frozen is not None:
            tid = getattr(trial_frozen, "_trial_id", None) or getattr(trial_frozen, "trial_id", None) or getattr(trial_frozen, "id", None)
    except Exception:
        pass

    # 如果仍然没 tid，尝试在 study.trials 中找到对应 frozen trial
    if tid is None:
        try:
            for ft in study.trials:
                if getattr(ft, "number", None) == mapped_number:
                    trial_frozen = ft
                    tid = getattr(ft, "_trial_id", None) or getattr(ft, "trial_id", None) or getattr(ft, "id", None)
                    break
        except Exception:
            pass

    if tid is None:
        print("safe_storage_write: 无法解析内部 trial id，无法用 storage 直接写入")
        return False

    # 尝试写入（多次重试）
    for attempt in range(tries):
        try:
            if hasattr(storage, "set_trial_value"):
                try:
                    storage.set_trial_value(tid, float(value))
                except Exception as e:
                    print(f"storage.set_trial_value 失败: {e}")
            if hasattr(storage, "set_trial_state"):
                try:
                    storage.set_trial_state(tid, state)
                except Exception as e:
                    print(f"storage.set_trial_state 失败: {e}")
            # 如果 storage 提供 set_trial_user_attr we could set fs_trial but not necessary
        except Exception as e:
            print(f"storage 写入尝试 {attempt} 失败: {e}")
        # reload study and verify
        try:
            study_reloaded = optuna.create_study(study_name=study.study_name, storage=f"sqlite:///{db_path}", load_if_exists=True)
            refreshed = next((t for t in study_reloaded.trials if t.number == mapped_number), None)
            if refreshed is not None and refreshed.state in (TrialState.COMPLETE, TrialState.PRUNED) and refreshed.value is not None:
                # success
                try:
                    refresh_fs_to_optuna(study_reloaded)
                except Exception:
                    pass
                print(f"storage 写入并核验成功：optuna#{mapped_number} state={refreshed.state}, value={refreshed.value}")
                return True
        except Exception:
            pass
        time.sleep(sleep_between)
    print("safe_storage_write: 多次尝试后仍未能核验写回")
    return False

# ===========================
# finalize_fs_trial：统一负责把 fs_trial 的结果写回 DB 并核验
# - 会多次尝试 study.tell（本地、重载后）；
# - 若仍失败，则调用 storage 回退写入并核验；
# - 写回成功后会 reload study 并 refresh_fs_to_optuna，确保内存映射最新。
# ===========================
def finalize_fs_trial(study, fs_num, candidate_optuna_number, value, state=TrialState.COMPLETE, max_retries=3):
    # 强制刷新映射
    try:
        refresh_fs_to_optuna(study)
    except Exception:
        pass

    mapped_id = FS_TO_OPTUNA.get(fs_num, None)
    # 额外在 study.trials 里再查一次以防万一
    if mapped_id is None:
        for t in study.trials:
            try:
                fa = t.user_attrs.get("fs_trial", None)
            except Exception:
                fa = None
            if fa is not None:
                try:
                    if int(fa) == int(fs_num):
                        mapped_id = t.number
                        FS_TO_OPTUNA[int(fs_num)] = mapped_id
                        break
                except Exception:
                    continue

    # 如果不存在映射，尝试创建（ask -> set_user_attr）
    if mapped_id is None:
        print(f"[finalize] DB 中没有映射到 fs_trial {fs_num} 的 optuna trial，尝试新建并写回")
        try:
            new_trial = study.ask()
            try:
                new_trial.set_user_attr("fs_trial", int(fs_num))
            except Exception:
                pass
            FS_TO_OPTUNA[int(fs_num)] = new_trial.number
            mapped_id = new_trial.number
        except Exception as e:
            print(f"[finalize] ask() 创建新 trial 失败: {e}")
            mapped_id = candidate_optuna_number

    # 现在 mapped_id 可能为 None 或一个数字 - 以数字为准
    if mapped_id is None:
        print(f"[finalize] 无法确定要写回的 optuna trial id (fs_trial {fs_num})")
        return False

    # 多次尝试写回：优先直接用当前 study 的 OptunaTrial -> study.tell()
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            # 取出最新的 study.trials 中的 frozen，判断状态
            frozen = next((t for t in study.trials if t.number == mapped_id), None)
            # 若 frozen 存在并已终态则直接返回成功（可能别处已写回）
            if frozen is not None and frozen.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL):
                print(f"[finalize] optuna#{mapped_id} 已处于终态 ({frozen.state})，跳过写回")
                # reload & refresh to ensure mapping consistent
                try:
                    study = optuna.create_study(study_name=study.study_name, storage=f"sqlite:///{db_path}", load_if_exists=True)
                    refresh_fs_to_optuna(study)
                except Exception:
                    pass
                return True

            # 使用 OptunaTrial 去写回（优先）
            try:
                trial_db = OptunaTrial(study, mapped_id)
                study.tell(trial_db, float(value), state=state)
                # reload & verify
                try:
                    study_reloaded = optuna.create_study(study_name=study.study_name, storage=f"sqlite:///{db_path}", load_if_exists=True)
                    refreshed = next((t for t in study_reloaded.trials if t.number == mapped_id), None)
                    if refreshed is not None and refreshed.state in (TrialState.COMPLETE, TrialState.PRUNED) and refreshed.value is not None:
                        refresh_fs_to_optuna(study_reloaded)
                        print(f"[finalize] 成功将 fs_trial {fs_num} 写回到 optuna#{mapped_id}（state={refreshed.state}, value={refreshed.value}）")
                        return True
                    else:
                        print(f"[finalize] 写回后核验未通过 (attempt {attempt})，将重试。")
                        study = study_reloaded
                        time.sleep(0.2)
                        attempt += 0  # continue loop
                except Exception as e_reload:
                    print(f"[finalize] 写回后重载 study 以核验失败: {e_reload}")
                    # fallthrough to next attempt/storage-write
            except Exception as e_tell:
                print(f"[finalize] study.tell(optuna#{mapped_id}) 失败 (attempt {attempt}): {e_tell}")
                # 继续尝试重载 study 并重试
                try:
                    study = optuna.create_study(study_name=study.study_name, storage=f"sqlite:///{db_path}", load_if_exists=True)
                    refresh_fs_to_optuna(study)
                except Exception as e_r:
                    print(f"[finalize] 重新加载 study 失败: {e_r}")
                time.sleep(0.2)
        except Exception as ex:
            print(f"[finalize] 写回循环出错: {ex}")
            time.sleep(0.2)

    # 如果循环完仍未写回，则尝试使用 storage 强制写入并核验
    print(f"[finalize] study.tell 多次失败或核验未通过，尝试使用 storage 层强制写入 optuna#{mapped_id}")
    try:
        study_reloaded = optuna.create_study(study_name=study.study_name, storage=f"sqlite:///{db_path}", load_if_exists=True)
        ok = safe_storage_write_and_verify(study_reloaded, fs_num, mapped_id, value, state)
        if ok:
            try:
                # reload original study and refresh mapping
                study_final = optuna.create_study(study_name=study_reloaded.study_name, storage=f"sqlite:///{db_path}", load_if_exists=True)
                refresh_fs_to_optuna(study_final)
            except Exception:
                pass
            return True
        else:
            print(f"[finalize] storage 强制写入也未核验通过 (optuna#{mapped_id})")
            return False
    except Exception as e:
        print(f"[finalize] 在尝试 storage 写入时出错: {e}")
        return False

# ===========================
# clean_invalid_trials (保持原来行为)
# ===========================
def clean_invalid_trials():
    existing_trials = get_existing_trial_numbers()
    if not existing_trials:
        print("未发现任何有效試驗，从0开始")
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
    incomplete_candidates = []

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
        last_epoch, _ = get_last_epoch(trial_dir)
        val = calculate_best_score(trial_dir)
        if last_epoch >= EPOCHS_PER_TRIAL:
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
            incomplete_candidates.append((trial_num, params, int(last_epoch), float(val)))

    skipped_incomplete = []
    if incomplete_candidates:
        chosen = sorted(incomplete_candidates, key=lambda x: x[0])[0]
        trial_num, params, last_epoch_chosen, val_chosen = chosen
        try:
            frozen = ot_trial.create_trial(
                params=params,
                distributions=distributions,
                value=None,
                state=TrialState.RUNNING,
                user_attrs={"fs_trial": int(trial_num), "last_epoch": int(last_epoch_chosen)}
            )
            new_study.add_trial(frozen)
            imported += 1
            print(f"将未完成的 fs trial_{trial_num} 导入为 RUNNING（优先恢复此 trial）。")
        except Exception as e:
            print(f"将未完成 trial_{trial_num} 导入为 RUNNING 时出错: {e}")
        for other in incomplete_candidates:
            if other[0] != trial_num:
                skipped_incomplete.append(other[0])

    print(f"已导入 {imported} 个有效試驗（包括最多 1 个 RUNNING）（≤ trial_{max_valid_trial}）")
    if skipped_incomplete:
        print(f"跳过导入 {len(skipped_incomplete)} 个未完成試驗（示例: {skipped_incomplete[:5]}），它们将在 resume 时按需继续。")

    try:
        if hasattr(new_storage, "engine") and new_storage.engine is not None:
            new_storage.engine.dispose()
    except Exception:
        pass

    try:
        backup = work_dir / f"study.db.bak_{timestamp}"
        if db_path.exists():
            try:
                db_path.replace(backup)
            except Exception:
                try:
                    shutil.copy2(db_path, backup)
                    safe_unlink(db_path)
                except Exception:
                    pass
        if tmp_db.exists():
            tmp_db.rename(db_path)
    except Exception as e:
        print(f"替换 study.db 时出错: {e}")
        try:
            if backup.exists() and not db_path.exists():
                backup.rename(db_path)
        except Exception:
            pass

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
# 参数采样（ask 流程）
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
# objective：运行训练并返回最终 best_score（不直接写 DB）
# ===========================
def objective(trial, fs_trial_num=None):
    fs_num = fs_trial_num if fs_trial_num is not None else trial.number
    optuna_display = FS_TO_OPTUNA.get(fs_num, getattr(trial, "number", None))
    trial_dir = work_dir / f"trial_{fs_num}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    params_file = trial_dir / "params.yaml"
    if params_file.exists():
        try:
            with open(params_file, 'r') as f:
                params = yaml.safe_load(f) or {}
        except Exception:
            params = {}
    else:
        try:
            params = getattr(trial, "_params", None) or sample_params_with_trial(trial)
        except Exception:
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
    update_trial_status(fs_num, EPOCHS_PER_TRIAL, "completed")
    print(f"[fs trial {fs_num} | optuna trial {optuna_display}] 完成，最优加权得分: {best_score:.4f}")
    return best_score

# ===========================
# 信号处理
# ===========================
def signal_handler(sig, frame):
    trial_status = load_trial_status()
    for k, info in trial_status.items():
        if info["status"] == "running":
            info["status"] = "interrupted"
    save_trial_status(trial_status)
    print("Ctrl+C detected. Trial statuses saved. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ===========================
# 主流程（在写回点统一调用 finalize_fs_trial）
# ===========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=TRIALS_TO_RUN)
    parser.add_argument("--epochs", type=int, default=EPOCHS_PER_TRIAL)
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-trial", type=int, nargs="*", help="要恢复的trial编号(可多个)")
    parser.add_argument("--clean", action="store_true", help="重建并替换 study.db（小心使用）")
    args = parser.parse_args()

    need_clean = args.clean or (not db_path.exists())
    max_valid_trial = -1
    if need_clean:
        max_valid_trial = clean_invalid_trials()
        print("DB 清理已完成（重建并只导入文件系统上存在的试验）。程序将直接退出，后续的 resume/继续运行请使用 --resume 或 --resume-trial 参数。")
        sys.exit(0)
    else:
        existing_trials = get_existing_trial_numbers()
        max_valid_trial = max(existing_trials) if existing_trials else -1

    TRIALS_TO_RUN = args.trials
    EPOCHS_PER_TRIAL = args.epochs

    sampler = TPESampler(seed=seed, n_startup_trials=30)
    pruner = MedianPruner(n_startup_trials=30) if args.prune else optuna.pruners.NopPruner()

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

    # 初始构造 fs->optuna 映射（从 DB）
    try:
        refresh_fs_to_optuna(study)
    except Exception:
        pass

    # ---- resume specific trials（指定恢复） ----
    if args.resume_trial:
        existing_trials = get_existing_trial_numbers()
        for trial_num in args.resume_trial:
            if trial_num not in existing_trials:
                print(f"跳过不存在的試驗: trial_{trial_num}")
                continue
            trial_obj = None
            refresh_fs_to_optuna(study)
            if trial_num in FS_TO_OPTUNA:
                optuna_id = FS_TO_OPTUNA[trial_num]
                mapped = next((t for t in study.trials if t.number == optuna_id), None)
                if mapped is not None and mapped.state not in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL):
                    try:
                        trial_obj = OptunaTrial(study, optuna_id)
                        print(f"复用已有的 optuna trial {optuna_id}（继续未完成試驗） -> 对应文件夹 trial_{trial_num} ...")
                    except Exception as e:
                        trial_obj = None
                        print(f"尝试复用 optuna trial {optuna_id} 失败（回退到 ask）：{e}")
            if trial_obj is None:
                params_file = work_dir / f"trial_{trial_num}" / "params.yaml"
                params = {}
                if params_file.exists():
                    try:
                        with open(params_file, 'r') as f:
                            params = yaml.safe_load(f) or {}
                    except Exception:
                        params = {}
                trial_obj = study.ask()
                try:
                    trial_obj.set_user_attr("fs_trial", int(trial_num))
                    FS_TO_OPTUNA[int(trial_num)] = trial_obj.number
                    if params:
                        try:
                            trial_obj._params = params
                            trial_obj._distributions = build_distributions_from_space()
                        except Exception:
                            pass
                except Exception:
                    pass
                print(f"恢复 fs trial {trial_num} -> optuna trial {trial_obj.number}")
            try:
                val = objective(trial_obj, fs_trial_num=trial_num)
                ok = finalize_fs_trial(study, trial_num, getattr(trial_obj, "number", None), float(val), state=TrialState.COMPLETE)
                if not ok:
                    print(f"警告：无法将试验结果写回 DB (fs trial {trial_num}).")
            except optuna.TrialPruned:
                trial_dir = work_dir / f"trial_{trial_num}"
                best = calculate_best_score(trial_dir)
                ok = finalize_fs_trial(study, trial_num, getattr(trial_obj, "number", None), float(best), state=TrialState.PRUNED)
                if not ok:
                    print(f"警告：剪枝结果写回失败 (fs trial {trial_num}).")
            except Exception as e:
                print(f"恢复 trial {trial_num} 时出错: {e}")
                traceback.print_exc()
        print("已完成指定 --resume-trial 的恢复任务，程序退出。")
        sys.exit(0)

    # ---- resume 所有未完成的 trials ----
    if args.resume:
        incomplete_set = set()
        try:
            for t in study.trials:
                try:
                    fa = t.user_attrs.get("fs_trial", None)
                except Exception:
                    fa = None
                if fa is not None:
                    try:
                        fs_num = int(fa)
                    except Exception:
                        continue
                    fs_dir = work_dir / f"trial_{fs_num}"
                    if fs_dir.exists() and t.state not in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL):
                        incomplete_set.add(fs_num)
        except Exception as e:
            print(f"从 DB 查找未完成试验时出错: {e}")

        try:
            status_incomplete = get_incomplete_trials_from_status()
            for n in status_incomplete:
                incomplete_set.add(n)
        except Exception:
            pass

        incomplete = sorted(list(incomplete_set))
        if incomplete:
            print(f"找到未完成的 trials (优先 DB 映射)：{incomplete}")
            for trial_num in incomplete:
                refresh_fs_to_optuna(study)
                trial_obj = None
                if trial_num in FS_TO_OPTUNA:
                    optuna_id = FS_TO_OPTUNA[trial_num]
                    mapped = next((t for t in study.trials if t.number == optuna_id), None)
                    if mapped is not None and mapped.state not in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL):
                        try:
                            trial_obj = OptunaTrial(study, optuna_id)
                            print(f"复用已有的 optuna trial {optuna_id}（继续未完成試驗） -> 对应文件夹 trial_{trial_num} ...")
                        except Exception as e:
                            trial_obj = None
                            print(f"尝试复用 optuna trial {optuna_id} 失败（回退到 ask）：{e}")
                if trial_obj is None:
                    params_file = work_dir / f"trial_{trial_num}" / "params.yaml"
                    params = {}
                    if params_file.exists():
                        try:
                            with open(params_file, 'r') as f:
                                params = yaml.safe_load(f) or {}
                        except Exception:
                            params = {}
                    trial_obj = study.ask()
                    try:
                        trial_obj.set_user_attr("fs_trial", int(trial_num))
                        FS_TO_OPTUNA[int(trial_num)] = trial_obj.number
                        if params:
                            try:
                                trial_obj._params = params
                                trial_obj._distributions = build_distributions_from_space()
                            except Exception:
                                pass
                    except Exception:
                        pass
                    print(f"恢复 fs trial {trial_num} -> optuna trial {trial_obj.number}")
                try:
                    val = objective(trial_obj, fs_trial_num=trial_num)
                    ok = finalize_fs_trial(study, trial_num, getattr(trial_obj, "number", None), float(val), state=TrialState.COMPLETE)
                    if not ok:
                        print(f"警告：无法将试验结果写回 DB (fs trial {trial_num}).")
                except optuna.TrialPruned:
                    best = calculate_best_score(work_dir / f"trial_{trial_num}")
                    ok = finalize_fs_trial(study, trial_num, getattr(trial_obj, "number", None), float(best), state=TrialState.PRUNED)
                    if not ok:
                        print(f"警告：剪枝写回失败: {trial_num}")
                except Exception as e:
                    print(f"恢复 trial {trial_num} 时出错: {e}")
                    traceback.print_exc()
        else:
            print("未找到基于 DB/状态文件的未完成试验，将按需要创建新 trials。")

    # 在 resume 完成后刷新映射，确保后续创建新 trial 不重复
    try:
        refresh_fs_to_optuna(study)
    except Exception:
        pass

    # ---- 创建新的 trials（补足到指定数量） ----
    existing_trials = get_existing_trial_numbers()
    existing_count = len(existing_trials)
    next_trials_needed = max(0, TRIALS_TO_RUN - existing_count)
    if next_trials_needed > 0:
        print(f"需要 {next_trials_needed} 个新 trials ...")
        start_trial_number = (max(existing_trials) + 1) if existing_trials else 0
        created = 0
        candidate = start_trial_number
        while created < next_trials_needed:
            existing_fs = set(get_existing_trial_numbers())
            while candidate in existing_fs:
                candidate += 1
            fs_trial_number = candidate
            refresh_fs_to_optuna(study)
            if fs_trial_number in FS_TO_OPTUNA:
                mapped_id = FS_TO_OPTUNA[fs_trial_number]
                fs_dir = work_dir / f"trial_{fs_trial_number}"
                if fs_dir.exists():
                    print(f"fs trial {fs_trial_number} 已在 DB 中映射到 optuna trial {mapped_id}，跳过创建。")
                    created += 1
                    candidate += 1
                    continue
                else:
                    print(f"警告：发现 DB 中对 fs trial {fs_trial_number} 的映射（optuna trial {mapped_id}），但文件夹不存在，正在移除旧映射。")
                    try:
                        t_obj = OptunaTrial(study, mapped_id)
                        t_obj.set_user_attr("fs_trial", None)
                    except Exception as e:
                        print(f"移除旧映射失败: {e}")
                    FS_TO_OPTUNA.pop(fs_trial_number, None)

            params_file = work_dir / f"trial_{fs_trial_number}" / "params.yaml"
            params = {}
            if params_file.exists():
                try:
                    with open(params_file, "r") as f:
                        params = yaml.safe_load(f) or {}
                except Exception:
                    params = {}
            trial_obj = study.ask()
            try:
                trial_obj.set_user_attr("fs_trial", int(fs_trial_number))
                FS_TO_OPTUNA[int(fs_trial_number)] = trial_obj.number
                if params:
                    try:
                        trial_obj._params = params
                        trial_obj._distributions = build_distributions_from_space()
                    except Exception:
                        pass
            except Exception:
                pass
            print(f"开始新 fs trial {fs_trial_number} (optuna trial {trial_obj.number}) ...")
            try:
                val = objective(trial_obj, fs_trial_num=fs_trial_number)
                ok = finalize_fs_trial(study, fs_trial_number, getattr(trial_obj, "number", None), float(val), state=TrialState.COMPLETE)
                if not ok:
                    print(f"警告：无法将新 trial 写回 DB (fs trial {fs_trial_number}).")
            except optuna.TrialPruned:
                best = calculate_best_score(work_dir / f"trial_{fs_trial_number}")
                ok = finalize_fs_trial(study, fs_trial_number, getattr(trial_obj, "number", None), float(best), state=TrialState.PRUNED)
                if not ok:
                    print(f"警告：剪枝写回失败: {fs_trial_number}")
            except Exception as e:
                print(f"运行 trial {fs_trial_number} 时出错: {e}")
                traceback.print_exc()
            created += 1
            candidate += 1

    print("=== 所有 trials 完成（或已调度） ===")

    # reload study 并打印 best
    try:
        study = optuna.create_study(
            study_name="yolo_optuna",
            direction="maximize",
            storage=storage_url,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )
    except Exception:
        pass
    try:
        refresh_fs_to_optuna(study)
    except Exception:
        pass
    try:
        if study.best_trial:
            print(f"最佳 optuna trial: {study.best_trial.number}, 分数={study.best_trial.value}")
            print(f"最佳参数: {study.best_trial.params}")
        else:
            print("没有找到最佳试验")
    except Exception as e:
        print(f"获取最佳 trial 失败: {e}")
