from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoConfig

from bearing.dataset_bearing import BearingRULEvalDataset, BearingRULIterableDataset
from bearing.models.chronosbolt_rul import ChronosBoltModelForRULWithRetrieval
from bearing.retrieve_bearing import RetrieverForRUL
from bearing.utils import read_yaml, regression_metrics, save_json, set_seed


@torch.no_grad()
def evaluate(model, loader, retriever, device):
    model.eval()
    records = []

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

        pred = outputs.rul_pred.detach().cpu().numpy().reshape(-1)
        target = batch['y'].numpy().reshape(-1)

        bearing_ids = batch['bearing_id']
        end_idx = batch['end_idx'].numpy().reshape(-1)
        y_steps = batch['y_steps'].numpy().reshape(-1)
        num_files = batch['num_files'].numpy().reshape(-1)

        pred_steps = pred * (num_files - 1.0)

        for i in range(len(pred)):
            records.append({
                'bearing_id': str(bearing_ids[i]),
                'end_idx': int(end_idx[i]),
                'y_true_norm': float(target[i]),
                'y_pred_norm': float(pred[i]),
                'y_true_steps': float(y_steps[i]),
                'y_pred_steps': float(pred_steps[i]),
            })

    import pandas as pd
    df = pd.DataFrame(records)

    overall_norm = regression_metrics(df['y_true_norm'].to_numpy(), df['y_pred_norm'].to_numpy())
    overall_steps = regression_metrics(df['y_true_steps'].to_numpy(), df['y_pred_steps'].to_numpy())

    per_bearing_rows = []
    for bearing_id, g in df.groupby('bearing_id'):
        per_bearing_rows.append({
            'bearing_id': bearing_id,
            'rmse_norm': regression_metrics(g['y_true_norm'].to_numpy(), g['y_pred_norm'].to_numpy())['rmse'],
            'mae_norm': regression_metrics(g['y_true_norm'].to_numpy(), g['y_pred_norm'].to_numpy())['mae'],
            'phm_norm': regression_metrics(g['y_true_norm'].to_numpy(), g['y_pred_norm'].to_numpy())['phm_score'],
            'rmse_steps': regression_metrics(g['y_true_steps'].to_numpy(), g['y_pred_steps'].to_numpy())['rmse'],
            'mae_steps': regression_metrics(g['y_true_steps'].to_numpy(), g['y_pred_steps'].to_numpy())['mae'],
            'phm_steps': regression_metrics(g['y_true_steps'].to_numpy(), g['y_pred_steps'].to_numpy())['phm_score'],
            'num_samples': int(len(g)),
        })

    per_bearing_df = pd.DataFrame(per_bearing_rows).sort_values('bearing_id').reset_index(drop=True)

    return {
        'overall_norm': overall_norm,
        'overall_steps': overall_steps,
        'per_bearing': per_bearing_df,
        'predictions': df,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description='Train RUL head with TS-RAG backbone')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    set_seed(int(cfg['runtime'].get('seed', 2021)))
    device = torch.device(cfg['runtime'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    processed_dir = Path(cfg['dataset']['processed_dir']).expanduser().resolve()
    db_path = Path(cfg['retrieval']['database_dir']).expanduser().resolve() / cfg['retrieval'].get('database_name', 'phm2012_rul_retrieval_database.parquet')
    import json
    meta = json.loads((db_path.parent / 'database_meta.json').read_text(encoding='utf-8'))
    retriever = RetrieverForRUL(db_path, dimension=int(meta['embedding_dim']))
    retriever.build_index()

    train_dataset = BearingRULIterableDataset(processed_dir / 'windows_train_retrieved.parquet', mode='train', top_k=cfg['retrieval']['top_k']).shuffle(cfg['training'].get('shuffle_buffer_length', 4096))
    val_dataset = BearingRULEvalDataset(processed_dir / 'windows_val_retrieved.parquet', top_k=cfg['retrieval']['top_k'])
    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg['training']['eval_batch_size'], shuffle=False, num_workers=0)

    config = AutoConfig.from_pretrained(cfg['model']['pretrained_model_path'])
    model = ChronosBoltModelForRULWithRetrieval.from_pretrained(
        cfg['model']['pretrained_model_path'],
        config=config,
        augment=cfg['model'].get('augment_mode', 'moe'),
    )
    base_weight = Path(cfg['model']['pretrained_model_path']) / 'autogluon_model.pth'
    if base_weight.exists():
        model.load_state_dict(torch.load(base_weight, map_location='cpu'), strict=False)
    freeze_backbone = bool(cfg['training'].get('freeze_backbone', True))
    if freeze_backbone:
        trainable_keywords = ['gate_layer', 'encode_mlp', 'mha', 'ffn', 'rul_head']
        for p in model.parameters():
            p.requires_grad = False
        for name, p in model.named_parameters():
            p.requires_grad = any(k in name for k in trainable_keywords)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['epochs'], eta_min=1e-6)

    out_dir = Path(cfg['training']['checkpoint_dir']).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    best_rmse = math.inf
    patience = int(cfg['training'].get('patience', 8))
    stale = 0

    steps_per_epoch = max(1, int(cfg['training'].get('steps_per_epoch', 200)))
    grad_clip = float(cfg['training'].get('grad_clip', 1.0))

    for epoch in range(1, int(cfg['training']['epochs']) + 1):
        model.train()
        losses = []
        for step, batch in enumerate(train_loader, start=1):
            if step > steps_per_epoch:
                break
            optimizer.zero_grad()
            retrieved_seqs = torch.tensor(retriever.whole_seq[batch['indices'].numpy()]).float().to(device)
            outputs = model(
                context=batch['x'].float().to(device),
                target=batch['y'].float().to(device),
                retrieved_seq=retrieved_seqs,
                distances=batch['distances'].float().to(device),
            )
            loss = outputs.loss
            loss.backward()
            clip_grad_norm_(params, grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))

        scheduler.step()
        val_result = evaluate(model, val_loader, retriever, device)
        val_metrics = val_result['overall_norm']
        train_loss = float(np.mean(losses)) if losses else float('nan')

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"val_rmse={val_metrics['rmse']:.6f} "
            f"val_mae={val_metrics['mae']:.6f} "
            f"val_phm={val_metrics['phm_score']:.6f}"
        )

        if val_metrics['rmse'] < best_rmse:
            best_rmse = val_metrics['rmse']
            stale = 0
            torch.save(model.state_dict(), out_dir / 'best_rul_model.pth')

            save_json({
                'epoch': epoch,
                'train_loss': train_loss,
                'overall_norm': val_result['overall_norm'],
                'overall_steps': val_result['overall_steps'],
            }, out_dir / 'best_metrics.json')

            val_result['per_bearing'].to_csv(out_dir / 'best_val_per_bearing_metrics.csv', index=False)
            val_result['predictions'].to_csv(out_dir / 'best_val_predictions.csv', index=False)
        else:
            stale += 1
            if stale >= patience:
                print('Early stopping triggered.')
                break

    print(f'Best validation overall RMSE (normalized): {best_rmse:.6f}')
    print(f'Best per-bearing validation metrics saved to: {out_dir / "best_val_per_bearing_metrics.csv"}')


if __name__ == '__main__':
    main()
