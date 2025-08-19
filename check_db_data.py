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
