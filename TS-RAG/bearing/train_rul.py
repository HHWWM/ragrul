"""
训练主脚本。
作用：
1. 加载检索数据库
2. 加载训练/验证数据集
3. 加载 Chronos-Bolt + RUL 头
4. 进行训练
5. 在验证集上选择最优模型

本版本额外支持一个最小对比实验开关：
- cfg["model"]["use_retrieval"] = True  -> 使用检索增强
- cfg["model"]["use_retrieval"] = False -> 不使用检索增强

这样你做对比实验时，只需要改一个配置变量，不用改其它流程。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoConfig

from bearing.dataset_bearing import BearingRULEvalDataset, BearingRULIterableDataset
from bearing.models.chronosbolt_rul import ChronosBoltModelForRULWithRetrieval
from bearing.retrieve_bearing import RetrieverForRUL
from bearing.utils import read_yaml, regression_metrics, save_json, set_seed


def use_retrieval_enabled(cfg) -> bool:
    """
    统一读取“是否使用检索增强”的开关。

    说明：
    - 如果配置文件里没有写 model.use_retrieval，就默认 True
    - 这样可以保持你原来仓库的行为不变
    """
    return bool(cfg.get("model", {}).get("use_retrieval", True))


@torch.no_grad()
def evaluate(model, loader, retriever, device, use_retrieval: bool = True):
    """
    在验证集上评估模型。

    同时输出两类指标：
    1. 归一化 RUL 指标
    2. 剩余快照步数指标

    参数
    ----
    model:
        当前模型
    loader:
        验证集 DataLoader
    retriever:
        检索器
    device:
        运行设备
    use_retrieval:
        是否使用检索增强
    """
    model.eval()

    records = []

    for batch in loader:
        # =========================================================
        # 对比实验核心逻辑：
        # True  -> 正常使用检索结果
        # False -> 完全不传检索结果
        # =========================================================
        if use_retrieval:
            retrieved_seqs = torch.tensor(
                retriever.whole_seq[batch["indices"].numpy()]
            ).float().to(device)
            distances = batch["distances"].float().to(device)
        else:
            retrieved_seqs = None
            distances = None

        outputs = model(
            context=batch["x"].float().to(device),
            target=None,
            retrieved_seq=retrieved_seqs,
            distances=distances,
            use_retrieval=use_retrieval,
        )

        pred = outputs.rul_pred.detach().cpu().numpy().reshape(-1)
        target = batch["y"].numpy().reshape(-1)
        bearing_ids = batch["bearing_id"]
        end_idx = batch["end_idx"].numpy().reshape(-1)
        y_steps = batch["y_steps"].numpy().reshape(-1)
        num_files = batch["num_files"].numpy().reshape(-1)

        # 把归一化预测值映射成“剩余快照步数”
        pred_steps = pred * (num_files - 1.0)

        for i in range(len(pred)):
            records.append({
                "bearing_id": str(bearing_ids[i]),
                "end_idx": int(end_idx[i]),
                "y_true_norm": float(target[i]),
                "y_pred_norm": float(pred[i]),
                "y_true_steps": float(y_steps[i]),
                "y_pred_steps": float(pred_steps[i]),
            })

    df = pd.DataFrame(records)

    overall_norm = regression_metrics(
        df["y_true_norm"].to_numpy(),
        df["y_pred_norm"].to_numpy()
    )
    overall_steps = regression_metrics(
        df["y_true_steps"].to_numpy(),
        df["y_pred_steps"].to_numpy()
    )

    per_bearing_rows = []
    for bearing_id, g in df.groupby("bearing_id"):
        m_norm = regression_metrics(
            g["y_true_norm"].to_numpy(),
            g["y_pred_norm"].to_numpy()
        )
        m_steps = regression_metrics(
            g["y_true_steps"].to_numpy(),
            g["y_pred_steps"].to_numpy()
        )

        per_bearing_rows.append({
            "bearing_id": bearing_id,
            "norm_mae": m_norm["mae"],
            "norm_rmse": m_norm["rmse"],
            "norm_phm": m_norm["phm_score"],
            "steps_mae": m_steps["mae"],
            "steps_rmse": m_steps["rmse"],
            "steps_phm": m_steps["phm_score"],
            "num_samples": int(len(g)),
        })

    per_bearing_df = pd.DataFrame(per_bearing_rows).sort_values("bearing_id").reset_index(drop=True)

    return {
        "overall_norm": overall_norm,
        "overall_steps": overall_steps,
        "per_bearing": per_bearing_df,
        "predictions": df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RUL head with TS-RAG backbone")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # =========================================================
    # 1. 读取配置
    # =========================================================
    cfg = read_yaml(args.config)
    use_retrieval = use_retrieval_enabled(cfg)
    print(f"[INFO] use_retrieval = {use_retrieval}")

    set_seed(int(cfg["runtime"].get("seed", 2021)))

    device = torch.device(
        cfg["runtime"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    processed_dir = Path(cfg["dataset"]["processed_dir"]).expanduser().resolve()

    # =========================================================
    # 2. 加载检索数据库
    #    注意：即便 use_retrieval=False，这里仍然保留 retriever 初始化，
    #    这样你原来的数据流、代码结构变化最小。
    # =========================================================
    db_path = (
        Path(cfg["retrieval"]["database_dir"]).expanduser().resolve()
        / cfg["retrieval"].get("database_name", "phm2012_rul_retrieval_database.parquet")
    )

    meta = json.loads((db_path.parent / "database_meta.json").read_text(encoding="utf-8"))
    retriever = RetrieverForRUL(db_path, dimension=int(meta["embedding_dim"]))
    retriever.build_index()

    # =========================================================
    # 3. 加载数据集
    #    这里仍然沿用你原来的 retrieved parquet，
    #    这样不需要改数据预处理和 parquet 结构。
    # =========================================================
    train_dataset = BearingRULIterableDataset(
        processed_dir / "windows_train_retrieved.parquet",
        mode="train",
        top_k=cfg["retrieval"]["top_k"],
    ).shuffle(cfg["training"].get("shuffle_buffer_length", 4096))

    val_dataset = BearingRULEvalDataset(
        processed_dir / "windows_val_retrieved.parquet",
        top_k=cfg["retrieval"]["top_k"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["eval_batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # =========================================================
    # 4. 加载预训练主干 + RUL 模型
    # =========================================================
    config = AutoConfig.from_pretrained(cfg["model"]["pretrained_model_path"])
    model = ChronosBoltModelForRULWithRetrieval.from_pretrained(
        cfg["model"]["pretrained_model_path"],
        config=config,
        augment=cfg["model"].get("augment_mode", "moe"),
    )

    # 如果存在额外基座权重，就加载
    base_weight = Path(cfg["model"]["pretrained_model_path"]) / "autogluon_model.pth"
    if base_weight.exists():
        model.load_state_dict(torch.load(base_weight, map_location="cpu"), strict=False)

    # =========================================================
    # 5. 可选冻结主干，只训练检索融合层和 RUL 头
    # =========================================================
    freeze_backbone = bool(cfg["training"].get("freeze_backbone", True))
    if freeze_backbone:
        trainable_keywords = ["gate_layer", "encode_mlp", "mha", "ffn", "rul_head"]

        for p in model.parameters():
            p.requires_grad = False

        for name, p in model.named_parameters():
            p.requires_grad = any(k in name for k in trainable_keywords)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=1e-6,
    )

    # =========================================================
    # 6. 训练相关输出目录
    # =========================================================
    out_dir = Path(cfg["training"]["checkpoint_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    best_rmse = math.inf
    patience = int(cfg["training"].get("patience", 8))
    stale = 0

    steps_per_epoch = max(1, int(cfg["training"].get("steps_per_epoch", 200)))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))

    # =========================================================
    # 7. 训练循环
    # =========================================================
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        losses = []

        for step, batch in enumerate(train_loader, start=1):
            if step > steps_per_epoch:
                break

            optimizer.zero_grad()

            # =====================================================
            # 对比实验核心逻辑：
            # 是否把检索结果喂给模型
            # =====================================================
            if use_retrieval:
                retrieved_seqs = torch.tensor(
                    retriever.whole_seq[batch["indices"].numpy()]
                ).float().to(device)
                distances = batch["distances"].float().to(device)
            else:
                retrieved_seqs = None
                distances = None

            outputs = model(
                context=batch["x"].float().to(device),
                target=batch["y"].float().to(device),
                retrieved_seq=retrieved_seqs,
                distances=distances,
                use_retrieval=use_retrieval,
            )

            loss = outputs.loss
            loss.backward()

            clip_grad_norm_(params, grad_clip)
            optimizer.step()

            losses.append(float(loss.item()))

        # =====================================================
        # 8. 每轮训练结束后做验证
        # =====================================================
        scheduler.step()

        val_result = evaluate(
            model=model,
            loader=val_loader,
            retriever=retriever,
            device=device,
            use_retrieval=use_retrieval,
        )

        val_metrics = val_result["overall_norm"]
        train_loss = float(np.mean(losses)) if losses else float("nan")

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"val_rmse={val_metrics['rmse']:.6f} "
            f"val_mae={val_metrics['mae']:.6f} "
            f"val_phm={val_metrics['phm_score']:.6f}"
        )

        # =====================================================
        # 9. 保存最优模型
        # =====================================================
        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            stale = 0

            torch.save(model.state_dict(), out_dir / "best_rul_model.pth")

            save_json(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "use_retrieval": use_retrieval,
                    "overall_norm": val_result["overall_norm"],
                    "overall_steps": val_result["overall_steps"],
                },
                out_dir / "best_metrics.json"
            )

            val_result["per_bearing"].to_csv(out_dir / "best_val_per_bearing_metrics.csv", index=False)
            val_result["predictions"].to_csv(out_dir / "best_val_predictions.csv", index=False)

        else:
            stale += 1
            if stale >= patience:
                print("Early stopping triggered.")
                break

    print(f"Best validation overall RMSE (normalized): {best_rmse:.6f}")
    print(f"Best per-bearing validation metrics saved to: {out_dir / 'best_val_per_bearing_metrics.csv'}")


if __name__ == "__main__":
    main()