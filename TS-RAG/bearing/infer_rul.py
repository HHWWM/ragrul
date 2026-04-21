from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig

from bearing.features import aggregate_window_features, extract_acc_features
from bearing.models.chronosbolt_rul import ChronosBoltModelForRULWithRetrieval
from bearing.retrieve_bearing import RetrieverForRUL, build_embedding_model, l2_normalize
from bearing.utils import read_yaml


ACC_COLS = (4, 5)


def load_window_from_bearing_dir(bearing_dir: Path, seq_len: int, fs: float, preprocess_artifacts: Dict, feature_cols: List[str]):
    files = sorted(bearing_dir.glob('acc_*.csv'))[-seq_len:]
    if len(files) < seq_len:
        raise ValueError(f'Need at least {seq_len} acc files, got {len(files)}')
    rows = []
    for path in files:
        df = pd.read_csv(path, header=None)
        h = df.iloc[:, ACC_COLS[0]].astype(np.float32).to_numpy()
        v = df.iloc[:, ACC_COLS[1]].astype(np.float32).to_numpy()
        rows.append(extract_acc_features(h, v, fs))
    feat_df = pd.DataFrame(rows)
    feat_df = feat_df[feature_cols]
    scaler = preprocess_artifacts['feature_scaler']
    pca = preprocess_artifacts['hi_pca']
    hi_scaler = preprocess_artifacts['hi_scaler']
    feat_scaled = scaler.transform(feat_df.to_numpy(dtype=np.float32))
    hi = pca.transform(feat_scaled).squeeze()
    hi = hi_scaler.transform(hi.reshape(-1, 1)).squeeze().astype(np.float32)
    query_features = aggregate_window_features(feat_scaled)
    return hi, query_features


def main() -> None:
    parser = argparse.ArgumentParser(description='One-shot RUL inference')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--bearing_dir', type=str, required=True)
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    device = torch.device(cfg['runtime'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    processed_dir = Path(cfg['dataset']['processed_dir']).expanduser().resolve()

    with (processed_dir / 'preprocess_artifacts.pkl').open('rb') as f:
        preprocess_artifacts = pickle.load(f)
    feature_cols = json.loads((processed_dir / 'feature_columns.json').read_text(encoding='utf-8'))

    seq_len = int(cfg['model']['seq_len'])
    fs = float(cfg['dataset'].get('sampling_rate_hz', 25600))
    x, query_features = load_window_from_bearing_dir(Path(args.bearing_dir), seq_len, fs, preprocess_artifacts, feature_cols)

    db_path = Path(cfg['retrieval']['database_dir']).expanduser().resolve() / cfg['retrieval'].get('database_name', 'phm2012_rul_retrieval_database.parquet')
    meta = json.loads((db_path.parent / 'database_meta.json').read_text(encoding='utf-8'))
    retriever = RetrieverForRUL(db_path, dimension=int(meta['embedding_dim']))
    retriever.build_index()

    embedding_model = build_embedding_model(cfg)
    if embedding_model is None:
        ts_embed = x.reshape(1, -1)
    else:
        embeds, _ = embedding_model.embed(torch.tensor(x[None, :], dtype=torch.float32))
        ts_embed = embeds[:, -1, :].float().cpu().numpy()
    query_vec = np.concatenate([
        l2_normalize(ts_embed.astype(np.float32)),
        float(cfg['retrieval'].get('feature_weight', 0.35)) * l2_normalize(query_features.reshape(1, -1).astype(np.float32))
    ], axis=1).astype(np.float32)
    indices, distances = retriever.search(query_vec, top_k=cfg['retrieval']['top_k'])
    retrieved_seqs = torch.tensor(retriever.whole_seq[indices]).float().to(device)

    config = AutoConfig.from_pretrained(cfg['model']['pretrained_model_path'])
    model = ChronosBoltModelForRULWithRetrieval.from_pretrained(
        cfg['model']['pretrained_model_path'], config=config, augment=cfg['model'].get('augment_mode', 'moe')
    )
    state = torch.load(Path(cfg['training']['checkpoint_dir']) / 'best_rul_model.pth', map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(
            context=torch.tensor(x[None, :]).float().to(device),
            target=None,
            retrieved_seq=retrieved_seqs,
            distances=torch.tensor(distances).float().to(device),
        )
    result = {
        'bearing_dir': str(Path(args.bearing_dir).resolve()),
        'pred_rul_norm': float(outputs.rul_pred.item()),
        'topk_distance_mean': float(np.mean(distances)),
        'topk_indices': indices[0].tolist(),
        'query_feature_dim': int(len(query_features)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
