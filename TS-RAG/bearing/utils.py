from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def read_yaml(path: str | Path) -> Dict:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 2021) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = 100.0 * (y_true - y_pred) / np.clip(y_true, 1e-6, None)
    out = np.where(
        err <= 0,
        np.exp(-np.log(0.5) * err / 5.0),
        np.exp(-np.log(0.5) * err / 20.0),
    )
    return float(np.mean(out))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean(np.square(y_true - y_pred))))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))))
    denom = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2, 'phm_score': phm_score(y_true, y_pred)}


def save_json(data: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def plot_bearing_curves(records, out_dir: str | Path, filename_prefix: str = 'bearing') -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_bearing = {}
    for r in records:
        by_bearing.setdefault(r['bearing_id'], []).append(r)
    for bearing_id, rows in by_bearing.items():
        rows = sorted(rows, key=lambda x: x['end_idx'])
        x = [r['end_idx'] for r in rows]
        y_true = [r['y_true_steps'] for r in rows]
        y_pred = [r['y_pred_steps'] for r in rows]
        plt.figure(figsize=(10, 4))
        plt.plot(x, y_true, label='true_rul_steps')
        plt.plot(x, y_pred, label='pred_rul_steps')
        plt.xlabel('window_end_index')
        plt.ylabel('RUL (steps)')
        plt.title(f'{bearing_id} RUL prediction')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f'{filename_prefix}_{bearing_id}.png', dpi=150)
        plt.close()
