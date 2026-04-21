from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm


class RetrieverForRUL:
    def __init__(self, retrieval_database_path: str | Path, dimension: int):
        self.retrieval_database_path = str(retrieval_database_path)
        self.dimension = int(dimension)
        self.index: Optional[faiss.IndexFlatL2] = None
        self.x = None
        self.y = None
        self.whole_seq = None
        self.window_ids = None
        self.bearing_ids = None
        self.query_features = None
        self.rul_targets = None

    def build_index(self) -> None:
        database = pd.read_parquet(self.retrieval_database_path)
        embeddings = np.vstack(database['embedding'].to_numpy()).astype('float32')
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        self.x = np.array(database['x'].tolist(), dtype=object)
        self.y = np.array(database['y_seq'].tolist(), dtype=object)
        self.whole_seq = np.array([np.concatenate([np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)]) for x, y in zip(self.x, self.y)], dtype=np.float32)
        self.window_ids = database['window_id'].to_numpy()
        self.bearing_ids = database['bearing_id'].to_numpy()
        self.query_features = np.array(database['query_features'].tolist(), dtype=np.float32)
        self.rul_targets = database['y_rul_norm'].to_numpy(dtype=np.float32)

    def search(self, query_vector: np.ndarray, top_k: int, params=None) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError('Index not built.')
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        if params is None:
            distances, indices = self.index.search(query_vector.astype('float32'), top_k + 1)
        else:
            distances, indices = self.index.search(query_vector.astype('float32'), top_k + 1, params=params)
        mask = distances[:, 0] == 0
        distances = np.where(mask[:, None], distances[:, 1:], distances[:, :-1])
        indices = np.where(mask[:, None], indices[:, 1:], indices[:, :-1])
        return indices.astype(np.int64), distances.astype(np.float32)


def read_yaml(path: Path) -> Dict:
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


from chronos import BaseChronosPipeline

def build_embedding_model(cfg):
    retrieval_cfg = cfg.get('retrieval', {})
    runtime_cfg = cfg.get('runtime', {})

    model_name = (
        retrieval_cfg.get('foundation_model_name')
        or retrieval_cfg.get('embedding_model_name')
        or retrieval_cfg.get('model_name')
    )
    if model_name is None:
        raise KeyError("retrieval.foundation_model_name / embedding_model_name / model_name not found in config")

    device_map = retrieval_cfg.get('device_map', runtime_cfg.get('device_map', 'auto'))
    dtype_name = retrieval_cfg.get('torch_dtype', runtime_cfg.get('embedding_dtype', 'bfloat16'))

    if isinstance(dtype_name, str):
        if dtype_name == 'auto':
            torch_dtype = 'auto'
        else:
            torch_dtype = getattr(torch, dtype_name)
    else:
        torch_dtype = dtype_name

    print(f"[INFO] loading Chronos model: {model_name}")
    print(f"[INFO] device_map={device_map}, torch_dtype={torch_dtype}")

    pipeline = BaseChronosPipeline.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )

    print(f"[INFO] pipeline type: {type(pipeline)}")
    print(f"[INFO] has embed: {hasattr(pipeline, 'embed')}")
    return pipeline


def embed_sequences(sequences: np.ndarray, embedding_model, batch_size: int) -> np.ndarray:
    vectors = []
    for start in tqdm(range(0, len(sequences), batch_size), desc='Embedding windows'):
        batch = np.stack(sequences[start:start + batch_size]).astype(np.float32)

        if embedding_model is None:
            vectors.append(batch)
        else:
            with torch.no_grad():
                context = torch.tensor(batch)
                embeds, _ = embedding_model.embed(context)
                last_token = embeds[:, -1, :].float().cpu().numpy()
                vectors.append(last_token)

    out = np.concatenate(vectors, axis=0)
    print("[INFO] sequence embedding shape:", out.shape)
    return out


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def create_retrieval_database(cfg: Dict) -> Path:
    processed_dir = Path(cfg['dataset']['processed_dir']).expanduser().resolve()
    train_windows = pd.read_parquet(processed_dir / 'windows_train.parquet')
    batch_size = int(cfg['retrieval'].get('embedding_batch_size', 256))
    feature_weight = float(cfg['retrieval'].get('feature_weight', 0.35))

    sequences = np.array(train_windows['x'].tolist(), dtype=np.float32)
    query_features = np.array(train_windows['query_features'].tolist(), dtype=np.float32)
    embedding_model = build_embedding_model(cfg)
    ts_embeds = embed_sequences(sequences, embedding_model, batch_size=batch_size)
    ts_embeds = l2_normalize(ts_embeds.astype(np.float32))
    query_features = l2_normalize(query_features.astype(np.float32))
    combined = np.concatenate([ts_embeds, feature_weight * query_features], axis=1).astype(np.float32)

    database = train_windows.copy()
    database['window_id'] = np.arange(len(database), dtype=np.int64)
    database['embedding'] = [row for row in combined]

    out_dir = Path(cfg['retrieval']['database_dir']).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / cfg['retrieval'].get('database_name', 'phm2012_rul_retrieval_database.parquet')
    database.to_parquet(db_path, index=False)

    meta = {
        'embedding_dim': int(combined.shape[1]),
        'num_entries': int(len(database)),
        'feature_weight': feature_weight,
    }
    with (out_dir / 'database_meta.json').open('w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return db_path


def attach_retrieval_results(cfg: Dict, db_path: Path) -> None:
    processed_dir = Path(cfg['dataset']['processed_dir']).expanduser().resolve()
    meta_path = db_path.parent / 'database_meta.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    retriever = RetrieverForRUL(db_path, dimension=int(meta['embedding_dim']))
    retriever.build_index()

    feature_weight = float(cfg['retrieval'].get('feature_weight', 0.35))
    embedding_model = build_embedding_model(cfg)
    batch_size = int(cfg['retrieval'].get('embedding_batch_size', 256))
    top_k = int(cfg['retrieval'].get('top_k', 5))

    for phase in ['train', 'val', 'test']:
        phase_path = processed_dir / f'windows_{phase}.parquet'
        if not phase_path.exists():
            continue
        df = pd.read_parquet(phase_path)
        sequences = np.array(df['x'].tolist(), dtype=np.float32)
        query_features = np.array(df['query_features'].tolist(), dtype=np.float32)
        ts_embeds = embed_sequences(sequences, embedding_model, batch_size=batch_size)
        combined = np.concatenate([
            l2_normalize(ts_embeds.astype(np.float32)),
            feature_weight * l2_normalize(query_features.astype(np.float32)),
        ], axis=1).astype(np.float32)
        indices, distances = retriever.search(combined, top_k=top_k)
        df['indices'] = [row.tolist() for row in indices]
        df['distances'] = [row.tolist() for row in distances]
        df.to_parquet(processed_dir / f'windows_{phase}_retrieved.parquet', index=False)
        print(f'Saved retrieved windows: {processed_dir / f"windows_{phase}_retrieved.parquet"}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Build PHM2012 retrieval database')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = read_yaml(Path(args.config))
    db_path = create_retrieval_database(cfg)
    attach_retrieval_results(cfg, db_path)
    print(f'Retrieval database ready: {db_path}')


if __name__ == '__main__':
    main()
