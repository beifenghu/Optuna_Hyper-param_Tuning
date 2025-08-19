# # check_optuna_db.py
import optuna
from pathlib import Path
db = Path("optuna_temp/study.db")
if not db.exists():
    print("study.db 不存在")
else:
    storage = f"sqlite:///{db}"
    study = optuna.load_study(study_name="yolo_optuna", storage=storage)
    print("Optuna study loaded. #trials:", len(study.trials))
    for t in sorted(study.trials, key=lambda x: x.number):
        print(f"trial {t.number} | state={t.state} | value={t.value} | params={t.params}")

# map_fs_to_db.py
# import optuna
# from pathlib import Path
# import pandas as pd
# import yaml

# work_dir = Path("optuna_temp")
# db = work_dir / "study.db"
# storage = f"sqlite:///{db}"
# METRIC_WEIGHTS = {"metrics/mAP50(B)":0.4,"metrics/mAP50-95(B)":0.3,"metrics/precision(B)":0.15,"metrics/recall(B)":0.15}

# def calc_best(trial_dir):
#     f = trial_dir / "train" / "results.csv"
#     if not f.exists():
#         f = trial_dir / "results.csv"
#     if not f.exists():
#         return None
#     try:
#         df = pd.read_csv(f)
#         best = 0.0
#         for _, row in df.iterrows():
#             s = 0.0
#             for m,w in METRIC_WEIGHTS.items():
#                 if m in row:
#                     s += float(row[m]) * w
#             if s>best: best=s
#         return float(best)
#     except Exception:
#         return None

# if not db.exists():
#     print("study.db 不存在")
#     raise SystemExit

# study = optuna.load_study(study_name="yolo_optuna", storage=storage)
# db_trials = {t.number: {"value": t.value, "params": t.params, "user_attrs": t.user_attrs} for t in study.trials}
# print("DB 中的 trial 概览（number:value）:")
# for k,v in sorted(db_trials.items()):
#     print(f"  optuna_trial {k} -> value={v['value']} params_keys={list(v['params'].keys())} user_attrs={v['user_attrs']}")

# print("\n文件系统 trial 与 DB 匹配尝试：")
# for d in sorted([p for p in work_dir.glob("trial_*") if p.is_dir()], key=lambda x:int(x.name.split('_')[1])):
#     n = int(d.name.split("_")[1])
#     best = calc_best(d)
#     print(f" fs trial_{n} -> best_score={best}")
#     if best is None:
#         continue
#     # find matching db trials by value (float equality)
#     matches = [tn for tn, info in db_trials.items() if info["value"] is not None and abs(info["value"] - best) < 1e-8]
#     print(f"   matches optuna_trial(s): {matches}")
