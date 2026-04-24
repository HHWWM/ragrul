"""
整个项目的数据入口。它负责扫描 PHM2012 的目录，读取每个快照，提特征，做标准化、PCA、HI 构造，然后切成训练、验证、测试窗口
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bearing.features import aggregate_window_features, extract_acc_features


ACC_COLS = (4, 5) # 第 5、6 列


def read_yaml(path: Path) -> Dict:
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)

"""
    读取一个 acc_*.csv。
    离线预处理比较稳健，会尝试逗号、分号、制表符、空白等多种分隔形式。
"""
def load_acc_file(path):
    import pandas as pd
    import numpy as np

    # 依次尝试常见分隔符
    seps = [",", ";", "\t", r"\s+"]

    last_shape = None
    for sep in seps:
        try:
            df = pd.read_csv(
                path,
                header=None,
                sep=sep,
                engine="python" if sep in [";", "\t", r"\s+"] else None
            )

            # 去掉全空列
            df = df.dropna(axis=1, how="all")

            last_shape = df.shape
            if df.shape[1] >= 6:
                h = pd.to_numeric(df.iloc[:, 4], errors="coerce").astype(np.float32).to_numpy()
                v = pd.to_numeric(df.iloc[:, 5], errors="coerce").astype(np.float32).to_numpy()

                mask = np.isfinite(h) & np.isfinite(v)
                h = h[mask]
                v = v[mask]

                if len(h) > 0 and len(v) > 0:
                    return h, v
        except Exception:
            continue

    raise ValueError(f"Failed to parse {path}, last_shape={last_shape}")

"""
    扫描 Learning_set 或 Full_Test_Set 下面的每个轴承文件夹。
"""
def iter_bearing_dirs(dataset_root: Path, split_name: str) -> List[Path]:
    split_dir = dataset_root / split_name
    if not split_dir.exists():
        return []
    return sorted([p for p in split_dir.iterdir() if p.is_dir()])

"""
    每个 acc_*.csv 当成一个快照。
    对每个快照提取特征，并给它分配 snapshot_idx、rul_steps、rul_norm。
"""
def build_snapshot_table(bearing_dir, fs, split_name):
    from pathlib import Path
    import pandas as pd

    bearing_dir = Path(bearing_dir)

    files = sorted([
        p for p in bearing_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv" and p.name.startswith("acc_")
    ])

    if not files:
        print(f"[WARN] no acc csv files found in {bearing_dir}")
        return pd.DataFrame()

    valid_items = []
    for path in files:
        try:
            h, v = load_acc_file(path)
            if len(h) < 32 or len(v) < 32:
                print(f"[WARN] too short, skip: {path}")
                continue
            feats = extract_acc_features(h, v, fs)
            valid_items.append((path, feats))
        except Exception as e:
            print(f"[WARN] skip bad file: {path} -> {e}")
            continue

    if not valid_items:
        return pd.DataFrame()

    rows = []
    total = len(valid_items)
    for idx, (path, feats) in enumerate(valid_items):
        row = {
            "split_root": split_name,
            "bearing_id": bearing_dir.name,
            "file_path": str(path),
            "file_name": path.name,
            "snapshot_idx": idx,
            "rul_steps": total - idx - 1,# 还剩多少个快照
            "rul_norm": (total - idx - 1) / max(total - 1, 1),# 归一化到 0~1
        }
        row.update(feats)
        rows.append(row)

    return pd.DataFrame(rows)

"""
    把“单快照表”切成“窗口表”。
    一个窗口包含:
    x             -> 长度为 seq_len 的 HI 序列
    y_seq         -> 后面 prediction_length 个时刻的归一化 RUL 序列
    y_rul_norm    -> 当前窗口末端时刻的归一化 RUL
    y_rul_steps   -> 当前窗口末端时刻对应的剩余快照步数
    query_features-> 当前窗口聚合特征
"""
def build_window_rows(
    snapshot_df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
    pred_len: int,
    split_name: str,
    train_fraction: float,
) -> List[Dict]:
    rows: List[Dict] = []
    for bearing_id, g in snapshot_df.groupby('bearing_id'):
        g = g.sort_values('snapshot_idx').reset_index(drop=True)

        hi = g['hi'].to_numpy(dtype=np.float32)
        rul_norm = g['rul_norm'].to_numpy(dtype=np.float32)
        rul_steps = g['rul_steps'].to_numpy(dtype=np.float32)
        feat_mat = g[feature_cols].to_numpy(dtype=np.float32)

        total = len(g)
        end_limit = total - pred_len
        if end_limit < seq_len:
            continue

        # Learning_set 内再按时间顺序切 train / val
        cut_idx = int(end_limit * train_fraction)

        for end_idx in range(seq_len - 1, end_limit):
            start_idx = end_idx - seq_len + 1
            x = hi[start_idx:end_idx + 1]
            y_seq = rul_norm[end_idx + 1:end_idx + 1 + pred_len]
            query_feat = aggregate_window_features(feat_mat[start_idx:end_idx + 1])

            if split_name == 'Learning_set':
                phase = 'train' if end_idx < cut_idx else 'val'
            else:
                phase = 'test'

            rows.append({
                'phase': phase,
                'bearing_id': bearing_id,
                'split_root': split_name,
                'start_idx': int(start_idx),
                'end_idx': int(end_idx),
                'num_files': int(total),
                'x': x.tolist(),
                'y_seq': y_seq.tolist(),
                'y_rul_norm': float(rul_norm[end_idx]),
                'y_rul_steps': float(rul_steps[end_idx]),
                'query_features': query_feat.tolist(),
            })
    return rows

"""
    总流程:
    1. 扫描数据集
    2. 提取每个快照的统计特征
    3. 用训练集拟合标准化缩放器 StandardScaler
    4. 用训练集拟合 PCA，把多维特征压成 1 维 HI
    5. 根据 HI 构建窗口
    6. 保存窗口表和预处理工件
"""
def main() -> None:
    parser = argparse.ArgumentParser(description='PHM2012 preprocessing for TS-RAG RUL')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = read_yaml(Path(args.config))
    dataset_root = Path(cfg['dataset']['root']).expanduser().resolve()
    out_root = Path(cfg['dataset']['processed_dir']).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    fs = float(cfg['dataset'].get('sampling_rate_hz', 25600))
    seq_len = int(cfg['model']['seq_len'])
    pred_len = int(cfg['model']['prediction_length'])
    train_fraction = float(cfg['dataset'].get('train_fraction_within_learning', 0.85))
# 1. 扫描所有 split
    snapshot_tables = []
    for split_name in cfg['dataset'].get('splits_to_scan', ['Learning_set', 'Full_Test_Set']):
        for bearing_dir in iter_bearing_dirs(dataset_root, split_name):
            table = build_snapshot_table(bearing_dir, fs=fs, split_name=split_name)
            if not table.empty:
                snapshot_tables.append(table)

    if not snapshot_tables:
        raise FileNotFoundError(f'No bearing data found under {dataset_root}')

    # ✅ 修复：移出 if 语句
    snapshot_df = pd.concat(snapshot_tables, axis=0, ignore_index=True)

    learning_mask = snapshot_df['split_root'].eq('Learning_set')
    if learning_mask.sum() == 0:
        raise RuntimeError('No Learning_set samples found for fitting scaler/PCA.')

    # 只保留数值列，彻底避免 file_path/file_name 进入特征矩阵
    numeric_df = snapshot_df.select_dtypes(include=[np.number]).copy()

    # 从数值列里排除标签/索引列
    drop_numeric_cols = {"snapshot_idx", "rul_steps", "rul_norm"}
    feature_cols = [c for c in numeric_df.columns if c not in drop_numeric_cols]

    print("dataset_root =", dataset_root)
    print("snapshot_df columns[:20] =", snapshot_df.columns[:20].tolist())
    print("numeric_df columns[:20] =", numeric_df.columns[:20].tolist())
    print("feature_cols[:10] =", feature_cols[:10])
    print("non_numeric_cols =", snapshot_df.select_dtypes(exclude=[np.number]).columns.tolist())
    print("sample file_path =", snapshot_df["file_path"].iloc[0] if "file_path" in snapshot_df.columns else "N/A")

    scaler = StandardScaler()

    X_train = numeric_df.loc[learning_mask, feature_cols].astype(np.float32).to_numpy()
    X_all = numeric_df.loc[:, feature_cols].astype(np.float32).to_numpy()

    scaler.fit(X_train)
    X_all_scaled = scaler.transform(X_all).astype(np.float32)

    # 写回标准化后的特征
    snapshot_df.loc[:, feature_cols] = X_all_scaled

    # PCA/HI 也只能在 Learning_set 上拟合
    X_train_scaled = snapshot_df.loc[learning_mask, feature_cols].astype(np.float32).to_numpy()
    X_all_scaled = snapshot_df.loc[:, feature_cols].astype(np.float32).to_numpy()
# 4. PCA -> HI
    pca = PCA(n_components=1)
    hi_train = pca.fit_transform(X_train_scaled).squeeze()

    train_rul = snapshot_df.loc[learning_mask, 'rul_norm'].to_numpy(dtype=np.float32)
    corr = np.corrcoef(hi_train, train_rul)[0, 1]

    full_hi = pca.transform(X_all_scaled).squeeze()
    if np.isfinite(corr) and corr > 0:
        full_hi = -full_hi

    hi_scaler = StandardScaler()
    hi_scaler.fit(full_hi[learning_mask.to_numpy()].reshape(-1, 1))
    snapshot_df['hi'] = hi_scaler.transform(full_hi.reshape(-1, 1)).squeeze().astype(np.float32)

    snapshot_df.to_parquet(out_root / 'snapshots_all.parquet', index=False)

    window_rows = []
    for split_name in cfg['dataset'].get('splits_to_scan', ['Learning_set', 'Full_Test_Set']):
        split_df = snapshot_df[snapshot_df['split_root'] == split_name].copy()
        if split_df.empty:
            continue
        window_rows.extend(build_window_rows(split_df, feature_cols, seq_len, pred_len, split_name, train_fraction))

    windows_df = pd.DataFrame(window_rows)
    if windows_df.empty:
        raise RuntimeError('No windows generated. Reduce seq_len/pred_len or check dataset.')

    for phase in ['train', 'val', 'test']:
        phase_df = windows_df[windows_df['phase'] == phase].reset_index(drop=True)
        if not phase_df.empty:
            phase_df.to_parquet(out_root / f'windows_{phase}.parquet', index=False)

    with (out_root / 'feature_columns.json').open('w', encoding='utf-8') as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    with (out_root / 'preprocess_artifacts.pkl').open('wb') as f:
        pickle.dump({'feature_scaler': scaler, 'hi_pca': pca, 'hi_scaler': hi_scaler}, f)

    metadata = {
        'dataset_root': str(dataset_root),
        'processed_dir': str(out_root),
        'sampling_rate_hz': fs,
        'seq_len': seq_len,
        'prediction_length': pred_len,
        'train_fraction_within_learning': train_fraction,
        'feature_dim': len(feature_cols),
        'window_feature_dim': len(windows_df.iloc[0]['query_features']),
    }
    with (out_root / 'metadata.json').open('w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        'snapshots': int(len(snapshot_df)),
        'train_windows': int((windows_df['phase'] == 'train').sum()),
        'val_windows': int((windows_df['phase'] == 'val').sum()),
        'test_windows': int((windows_df['phase'] == 'test').sum()),
        'processed_dir': str(out_root),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
