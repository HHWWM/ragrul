"""
负责离线评估和画曲线。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig

from bearing.dataset_bearing import BearingRULEvalDataset
from bearing.models.chronosbolt_rul import ChronosBoltModelForRULWithRetrieval
from bearing.retrieve_bearing import RetrieverForRUL
from bearing.utils import plot_bearing_curves, read_yaml, regression_metrics, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate RUL model')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    device = torch.device(cfg['runtime'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    processed_dir = Path(cfg['dataset']['processed_dir']).expanduser().resolve()
    eval_path = processed_dir / 'windows_test_retrieved.parquet'
    if not eval_path.exists():
        eval_path = processed_dir / 'windows_val_retrieved.parquet'

    dataset = BearingRULEvalDataset(eval_path, top_k=cfg['retrieval']['top_k'])
    loader = DataLoader(
        dataset,
        batch_size=cfg['training']['eval_batch_size'],
        shuffle=False,
        num_workers=0,
    )

    db_path = (
        Path(cfg['retrieval']['database_dir']).expanduser().resolve()
        / cfg['retrieval'].get('database_name', 'phm2012_rul_retrieval_database.parquet')
    )
    meta = json.loads((db_path.parent / 'database_meta.json').read_text(encoding='utf-8'))
    retriever = RetrieverForRUL(db_path, dimension=int(meta['embedding_dim']))
    retriever.build_index()

    config = AutoConfig.from_pretrained(cfg['model']['pretrained_model_path'])
    model = ChronosBoltModelForRULWithRetrieval.from_pretrained(
        cfg['model']['pretrained_model_path'],
        config=config,
        augment=cfg['model'].get('augment_mode', 'moe'),
    )
    state = torch.load(Path(cfg['training']['checkpoint_dir']) / 'best_rul_model.pth', map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    y_true_norm, y_pred_norm = [], []
    y_true_steps, y_pred_steps = [], []
    records = []

    with torch.no_grad():
        for batch in loader:
            retrieved_seqs = torch.tensor(
                retriever.whole_seq[batch['indices'].numpy()]
            ).float().to(device)

            outputs = model(
                context=batch['x'].float().to(device),
                target=None,
                retrieved_seq=retrieved_seqs,
                distances=batch['distances'].float().to(device),
            )

            pred_norm = outputs.rul_pred.detach().cpu().numpy().reshape(-1)
            true_norm = batch['y'].numpy().reshape(-1)

            max_steps = batch['num_files'].numpy().reshape(-1) - 1.0
            pred_steps = pred_norm * max_steps
            true_steps = batch['y_steps'].numpy().reshape(-1)

            end_idx = batch['end_idx'].numpy().reshape(-1)
            bearing_ids = batch['bearing_id']

            y_true_norm.extend(true_norm.tolist())
            y_pred_norm.extend(pred_norm.tolist())
            y_true_steps.extend(true_steps.tolist())
            y_pred_steps.extend(pred_steps.tolist())

            for i in range(len(pred_norm)):
                records.append({
                    'bearing_id': str(bearing_ids[i]),
                    'end_idx': int(end_idx[i]),
                    'y_true_norm': float(true_norm[i]),
                    'y_pred_norm': float(pred_norm[i]),
                    'y_true_steps': float(true_steps[i]),
                    'y_pred_steps': float(pred_steps[i]),
                    'num_files': int(max_steps[i] + 1),
                })

    metrics_norm = regression_metrics(np.array(y_true_norm), np.array(y_pred_norm))
    metrics_steps = regression_metrics(np.array(y_true_steps), np.array(y_pred_steps))

    out_dir = Path(cfg['evaluation']['output_dir']).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records_df = pd.DataFrame(records).sort_values(['bearing_id', 'end_idx']).reset_index(drop=True)

    per_bearing_rows = []
    for bearing_id, g in records_df.groupby('bearing_id'):
        m_norm = regression_metrics(g['y_true_norm'].to_numpy(), g['y_pred_norm'].to_numpy())
        m_steps = regression_metrics(g['y_true_steps'].to_numpy(), g['y_pred_steps'].to_numpy())

        per_bearing_rows.append({
            'bearing_id': bearing_id,
            'rmse_norm': m_norm['rmse'],
            'mae_norm': m_norm['mae'],
            'phm_norm': m_norm['phm_score'],
            'rmse_steps': m_steps['rmse'],
            'mae_steps': m_steps['mae'],
            'phm_steps': m_steps['phm_score'],
            'num_samples': int(len(g)),
        })

    per_bearing_df = pd.DataFrame(per_bearing_rows).sort_values('bearing_id').reset_index(drop=True)

    save_json(
        {
            'normalized': metrics_norm,
            'steps': metrics_steps,
        },
        out_dir / 'metrics.json'
    )

    records_df.to_csv(out_dir / 'all_test_predictions.csv', index=False)
    per_bearing_df.to_csv(out_dir / 'per_bearing_metrics.csv', index=False)

    plot_bearing_curves(records, out_dir / 'plots')

    print({
        'normalized': metrics_norm,
        'steps': metrics_steps,
        'per_bearing_metrics': str(out_dir / 'per_bearing_metrics.csv'),
        'all_predictions': str(out_dir / 'all_test_predictions.csv'),
        'output_dir': str(out_dir),
    })


if __name__ == '__main__':
    main()