"""
在原 TS-RAG 主干上接一个 RUL 回归头。检索增强、编码、融合逻辑不变，只改最后的任务头。
复用了原 ChronosBolt 的 patching / encoder / decoder / MoE/ARM 融合过程。
核心意思是:当前窗口编码 + 邻居未来片段编码 + 注意力 / 门控融合 -> 输出一个用于回归的表示。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config

from models.ChronosBolt import ChronosBoltModelForForecastingWithRetrieval


@dataclass
class RULModelOutput:
    loss: Optional[torch.Tensor] = None
    rul_pred: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None

"""
    在原来的“带检索增强的时间序列预测模型”基础上，
    换一个“归一化剩余寿命回归头”。
"""
class ChronosBoltModelForRULWithRetrieval(ChronosBoltModelForForecastingWithRetrieval):
    def __init__(self, config: T5Config, augment: str = 'moe'):
        super().__init__(config=config, augment=augment)
        self.rul_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.d_model // 2, 1),
        )
        self.loss_fn = nn.SmoothL1Loss(beta=0.05)
        self.init_extra_weights([self.rul_head])

    def forward(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        retrieved_seq: Optional[torch.Tensor] = None,
        distances: Optional[torch.Tensor] = None,
    ) -> RULModelOutput:
        """
            输入:
            context       -> 当前窗口的 x，也就是长度为 seq_len 的 HI 序列
            retrieved_seq -> 检索到的邻居 whole_seq = [邻居x, 邻居y_seq]
            distances     -> 邻居距离
            target        -> 当前窗口的 y_rul_norm
        """
        mask = mask.to(context.dtype) if mask is not None else torch.isnan(context).logical_not().to(context.dtype)
        batch_size, _ = context.shape
        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length:]
            mask = mask[..., -self.chronos_config.context_length:]

        context, _ = self.instance_norm(context)
        if retrieved_seq is None:
            raise ValueError('retrieved_seq is required for retrieval-augmented RUL prediction.')
        retrieved_seq, _ = self.instance_norm(retrieved_seq)

        if 'moe' not in self.augment:
            if distances is None:
                raise ValueError('distances are required when augment != moe')
            weights = torch.softmax(-distances, dim=1)
            retrieved_seq = (weights.unsqueeze(-1) * retrieved_seq).sum(dim=1, keepdim=True)

        rul_curve_len = self.chronos_config.prediction_length
        _, retrieved_k, retrieved_len = retrieved_seq.shape
        if retrieved_len < rul_curve_len + 1:
            raise ValueError('retrieved_seq length must be >= context + prediction_length')
        retrieved_x, retrieved_y = retrieved_seq.split((retrieved_len - rul_curve_len, rul_curve_len), dim=2)

        context = context.to(self.dtype)
        mask = mask.to(self.dtype)
        retrieved_y = retrieved_y.to(self.dtype)

        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        patched_context[~(patched_mask > 0)] = 0.0
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)
        attention_mask = patched_mask.sum(dim=-1) > 0
        input_embeds = self.input_patch_embedding(patched_context)

        if self.chronos_config.use_reg_token:
            reg_input_ids = torch.full((batch_size, 1), self.config.reg_token_id, device=input_embeds.device)
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat([attention_mask, torch.ones_like(reg_input_ids)], dim=-1)

        encoder_outputs = self.encoder(attention_mask=attention_mask, inputs_embeds=input_embeds)
        hidden_states = encoder_outputs[0]
        sequence_output = self.decode(input_embeds, attention_mask, hidden_states)

        if 'moe' in self.augment:
            retrieved_y_enc = []
            for i in range(retrieved_k):
                retrieved_y_enc.append(self.encode_mlp(retrieved_y[:, i, :]))
            retrieved_y_enc = torch.stack(retrieved_y_enc, dim=1)
            all_enc = torch.cat([sequence_output, retrieved_y_enc], dim=1)
            att_output, _ = self.mha(all_enc, all_enc, all_enc)
            att_output = all_enc + att_output
            att_output = att_output + self.dropout(self.ffn(att_output))
            scores = []
            for i in range(retrieved_k + 1):
                scores.append(torch.sigmoid(self.gate_layer(att_output[:, i, :])))
            scores = torch.stack(scores, dim=1)
            alpha = F.softmax(scores, dim=1)
            fused = torch.sum(alpha * att_output, dim=1)
            fused = self.dropout(fused)
            sequence_output = sequence_output + fused.unsqueeze(1)

        reg_features = sequence_output.squeeze(1)
        rul_pred = torch.sigmoid(self.rul_head(reg_features)).squeeze(-1)
        loss = None
        if target is not None:
            target = target.to(rul_pred.device).float().view(-1)
            loss = self.loss_fn(rul_pred, target)
        return RULModelOutput(loss=loss, rul_pred=rul_pred, features=reg_features)
