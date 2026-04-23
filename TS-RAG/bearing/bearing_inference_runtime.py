from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig

from bearing.features import aggregate_window_features, extract_acc_features
from bearing.models.chronosbolt_rul import ChronosBoltModelForRULWithRetrieval
from bearing.retrieve_bearing import RetrieverForRUL, build_embedding_model, l2_normalize
from bearing.utils import read_yaml

ACC_COLS = (4, 5)


def _resolve_env(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1], "")
    return value


def _resolve_device(requested: Optional[str]) -> torch.device:
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"

    requested = str(requested).lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


@dataclass
class RuntimeArtifacts:
    cfg: Dict[str, Any]
    device: torch.device
    processed_dir: Path
    db_path: Path
    feature_cols: List[str]
    preprocess_artifacts: Dict[str, Any]
    retriever: RetrieverForRUL
    embedding_model: Any
    model: ChronosBoltModelForRULWithRetrieval
    db_meta_df: pd.DataFrame


class BearingRULRuntime:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).expanduser().resolve()
        cfg = read_yaml(self.config_path)
        self.cfg = self._resolve_cfg(cfg)
        self.artifacts = self._load_artifacts(self.cfg)

    def _resolve_cfg(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        if "thingsboard" in cfg:
            cfg["thingsboard"] = {k: _resolve_env(v) for k, v in cfg["thingsboard"].items()}
        return cfg

    def _load_artifacts(self, cfg: Dict[str, Any]) -> RuntimeArtifacts:
        device = _resolve_device(cfg.get("runtime", {}).get("device"))

        processed_dir = Path(cfg["dataset"]["processed_dir"]).expanduser().resolve()
        db_path = (
            Path(cfg["retrieval"]["database_dir"]).expanduser().resolve()
            / cfg["retrieval"].get("database_name", "phm2012_rul_retrieval_database.parquet")
        )

        with (processed_dir / "preprocess_artifacts.pkl").open("rb") as f:
            preprocess_artifacts = pickle.load(f)

        feature_cols = json.loads((processed_dir / "feature_columns.json").read_text(encoding="utf-8"))

        meta = json.loads((db_path.parent / "database_meta.json").read_text(encoding="utf-8"))
        retriever = RetrieverForRUL(db_path, dimension=int(meta["embedding_dim"]))
        retriever.build_index()

        embedding_model = build_embedding_model(cfg)

        config = AutoConfig.from_pretrained(cfg["model"]["pretrained_model_path"])
        model = ChronosBoltModelForRULWithRetrieval.from_pretrained(
            cfg["model"]["pretrained_model_path"],
            config=config,
            augment=cfg["model"].get("augment_mode", "moe"),
        )

        ckpt_path = Path(cfg["training"]["checkpoint_dir"]).expanduser().resolve() / "best_rul_model.pth"
        state = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)

        if missing:
            print(f"[WARN] missing keys when loading checkpoint: {len(missing)}")
        if unexpected:
            print(f"[WARN] unexpected keys when loading checkpoint: {len(unexpected)}")

        model.to(device)
        model.eval()

        db_meta_df = pd.read_parquet(
            db_path,
            columns=["window_id", "bearing_id", "y_rul_norm", "y_rul_steps"],
        )

        return RuntimeArtifacts(
            cfg=cfg,
            device=device,
            processed_dir=processed_dir,
            db_path=db_path,
            feature_cols=feature_cols,
            preprocess_artifacts=preprocess_artifacts,
            retriever=retriever,
            embedding_model=embedding_model,
            model=model,
            db_meta_df=db_meta_df,
        )

    def get_all_acc_files(self, bearing_dir: str | Path) -> List[Path]:
        bearing_dir = Path(bearing_dir).expanduser().resolve()
        files = sorted(bearing_dir.glob("acc_*.csv"))
        return files

    def get_valid_window_range(self, bearing_dir: str | Path) -> Dict[str, int]:
        files = self.get_all_acc_files(bearing_dir)
        seq_len = int(self.cfg["model"]["seq_len"])
        total = len(files)

        return {
            "total_files": total,
            "seq_len": seq_len,
            "min_end_idx": seq_len,
            "max_end_idx": total,
        }

    def _load_window_from_bearing_dir(
        self,
        bearing_dir: Path,
        end_idx: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        seq_len = int(self.cfg["model"]["seq_len"])
        fs = float(self.cfg["dataset"].get("sampling_rate_hz", 25600))

        all_files = sorted(bearing_dir.glob("acc_*.csv"))
        total_files = len(all_files)

        if total_files < seq_len:
            raise ValueError(
                f"Need at least {seq_len} acc files, got {total_files} from {bearing_dir}"
            )

        if end_idx is None:
            end_idx = total_files

        end_idx = int(end_idx)
        end_idx = min(end_idx, total_files)
        end_idx = max(end_idx, seq_len)

        start_idx = end_idx - seq_len
        files = all_files[start_idx:end_idx]

        rows = []
        for path in files:
            df = pd.read_csv(path, header=None)
            h = df.iloc[:, ACC_COLS[0]].astype(np.float32).to_numpy()
            v = df.iloc[:, ACC_COLS[1]].astype(np.float32).to_numpy()
            rows.append(extract_acc_features(h, v, fs))

        feat_df = pd.DataFrame(rows)[self.artifacts.feature_cols]

        scaler = self.artifacts.preprocess_artifacts["feature_scaler"]
        pca = self.artifacts.preprocess_artifacts["hi_pca"]
        hi_scaler = self.artifacts.preprocess_artifacts["hi_scaler"]

        feat_scaled = scaler.transform(feat_df.to_numpy(dtype=np.float32))
        hi = pca.transform(feat_scaled).squeeze()
        hi = hi_scaler.transform(hi.reshape(-1, 1)).squeeze().astype(np.float32)
        query_features = aggregate_window_features(feat_scaled)

        return hi, query_features, start_idx, end_idx

    def _build_query_vector(self, x: np.ndarray, query_features: np.ndarray) -> np.ndarray:
        if self.artifacts.embedding_model is None:
            ts_embed = x.reshape(1, -1)
        else:
            with torch.no_grad():
                embeds, _ = self.artifacts.embedding_model.embed(
                    torch.tensor(x[None, :], dtype=torch.float32)
                )
            ts_embed = embeds[:, -1, :].float().cpu().numpy()

        feature_weight = float(self.cfg["retrieval"].get("feature_weight", 0.35))

        return np.concatenate(
            [
                l2_normalize(ts_embed.astype(np.float32)),
                feature_weight * l2_normalize(query_features.reshape(1, -1).astype(np.float32)),
            ],
            axis=1,
        ).astype(np.float32)

    def predict(self, bearing_dir: str | Path, end_idx: int | None = None) -> Dict[str, Any]:
        bearing_dir = Path(bearing_dir).expanduser().resolve()

        x, query_features, start_idx, actual_end_idx = self._load_window_from_bearing_dir(
            bearing_dir,
            end_idx=end_idx,
        )

        query_vec = self._build_query_vector(x, query_features)

        top_k = int(self.cfg["retrieval"]["top_k"])
        indices, distances = self.artifacts.retriever.search(query_vec, top_k=top_k)

        retrieved_seqs = torch.tensor(
            self.artifacts.retriever.whole_seq[indices]
        ).float().to(self.artifacts.device)

        with torch.no_grad():
            outputs = self.artifacts.model(
                context=torch.tensor(x[None, :]).float().to(self.artifacts.device),
                target=None,
                retrieved_seq=retrieved_seqs,
                distances=torch.tensor(distances).float().to(self.artifacts.device),
            )

        pred_rul_norm = float(outputs.rul_pred.item())

        matched = self.artifacts.db_meta_df.iloc[indices[0]].reset_index(drop=True)
        mean_neighbor_rul_norm = float(matched["y_rul_norm"].mean())
        mean_neighbor_rul_steps = float(matched["y_rul_steps"].mean())

        pred_rul_steps_proxy = None
        if abs(mean_neighbor_rul_norm) > 1e-6:
            pred_rul_steps_proxy = float(
                pred_rul_norm / mean_neighbor_rul_norm * mean_neighbor_rul_steps
            )

        return {
            "bearing_dir": str(bearing_dir),
            "bearing_id": bearing_dir.name,
            "pred_rul_norm": pred_rul_norm,
            "pred_rul_steps_proxy": pred_rul_steps_proxy,
            "topk_distance_mean": float(np.mean(distances)),
            "topk_distance_min": float(np.min(distances)),
            "topk_indices": indices[0].tolist(),
            "topk_neighbor_bearings": matched["bearing_id"].astype(str).tolist(),
            "topk_neighbor_rul_norm_mean": mean_neighbor_rul_norm,
            "topk_neighbor_rul_steps_mean": mean_neighbor_rul_steps,
            "query_feature_dim": int(len(query_features)),
            "seq_len": int(self.cfg["model"]["seq_len"]),
            "window_start_idx": int(start_idx),
            "window_end_idx": int(actual_end_idx),
            "window_size": int(actual_end_idx - start_idx),
            "total_files": int(len(self.get_all_acc_files(bearing_dir))),
        }


__all__ = ["BearingRULRuntime"]