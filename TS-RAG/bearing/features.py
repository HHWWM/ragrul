from __future__ import annotations

import math
from typing import Dict, Iterable, List

import numpy as np
import pywt
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew


EPS = 1e-12

"""
    安全计算熵，避免出现 0 的对数和除零问题。
"""
def _safe_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = np.abs(values)
    total = values.sum()
    if total <= EPS:
        return 0.0
    p = values / total
    p = p[p > 0]
    return float(-(p * np.log(p + EPS)).sum())
"""
    安全除法，避免分母接近 0。
"""
def _safe_div(a: float, b: float) -> float:
    return float(a / (b + EPS))

"""
    时域特征。
    这些特征反映振动信号在时域上的离散程度、冲击性、尖锐程度和能量水平。
"""
def time_domain_features(x: np.ndarray, prefix: str) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    abs_x = np.abs(x)
    mean = float(np.mean(x))
    std = float(np.std(x))
    rms = float(np.sqrt(np.mean(np.square(x)) + EPS))
    peak = float(np.max(abs_x))
    p2p = float(np.ptp(x))
    mean_abs = float(np.mean(abs_x))
    sqrt_amp = float(np.square(np.mean(np.sqrt(abs_x + EPS))))
    energy = float(np.sum(np.square(x)))
    feats = {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_var": float(np.var(x)),
        f"{prefix}_rms": rms,
        f"{prefix}_peak": peak,
        f"{prefix}_p2p": p2p,
        f"{prefix}_mean_abs": mean_abs,
        f"{prefix}_skew": float(skew(x, bias=False, nan_policy='omit')) if len(x) > 2 else 0.0,
        f"{prefix}_kurtosis": float(kurtosis(x, bias=False, nan_policy='omit')) if len(x) > 3 else 0.0,
        f"{prefix}_crest_factor": _safe_div(peak, rms),# 峰值 / 均方根
        f"{prefix}_impulse_factor": _safe_div(peak, mean_abs), # 峰值 / 平均绝对值
        f"{prefix}_shape_factor": _safe_div(rms, mean_abs),# 均方根 / 平均绝对值
        f"{prefix}_clearance_factor": _safe_div(peak, sqrt_amp),
        f"{prefix}_energy": energy,
        f"{prefix}_entropy": _safe_entropy(np.histogram(x, bins=64)[0]),
    }
    return feats

"""
    频域特征。
    把信号做频谱分析，描述能量分布在频率上的位置和扩散情况。
"""
def freq_domain_features(x: np.ndarray, fs: float, prefix: str) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    spec = np.abs(rfft(x))
    power = np.square(spec)
    freqs = rfftfreq(len(x), d=1.0 / fs)
    if len(power) == 0:
        return {f"{prefix}_spec_energy": 0.0}
    power_sum = float(power.sum() + EPS)
    centroid = float(np.sum(freqs * power) / power_sum)
    bw = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / power_sum))
    dominant_idx = int(np.argmax(power))
    cdf = np.cumsum(power) / power_sum
    rolloff_idx = int(np.searchsorted(cdf, 0.85))

    def band_energy(low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        return float(power[mask].sum() / power_sum)

    nyq = fs / 2.0
    feats = {
        f"{prefix}_spec_energy": float(power.sum()),
        f"{prefix}_spec_entropy": _safe_entropy(power),
        f"{prefix}_spectral_centroid": centroid,
        f"{prefix}_spectral_bandwidth": bw,
        f"{prefix}_dominant_freq": float(freqs[dominant_idx]),
        f"{prefix}_rolloff_85": float(freqs[min(rolloff_idx, len(freqs) - 1)]),
        f"{prefix}_band_0_20": band_energy(0.0, nyq * 0.2),
        f"{prefix}_band_20_50": band_energy(nyq * 0.2, nyq * 0.5),
        f"{prefix}_band_50_80": band_energy(nyq * 0.5, nyq * 0.8),
        f"{prefix}_band_80_100": band_energy(nyq * 0.8, nyq * 1.01),
    }
    return feats

"""
    时频域特征。
    用小波分解看不同尺度下的能量分布。
"""
def time_frequency_features(x: np.ndarray, prefix: str, wavelet: str = 'db4', level: int = 3) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    max_level = min(level, pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len))
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=max_level)
    energies = [float(np.sum(np.square(c))) for c in coeffs]
    total = sum(energies) + EPS
    feats: Dict[str, float] = {
        f"{prefix}_wavelet_entropy": _safe_entropy(np.array(energies)),
        f"{prefix}_wavelet_total_energy": float(total),
    }
    for i, e in enumerate(energies):
        feats[f"{prefix}_wavelet_e{i}"] = float(e / total)
    return feats

"""
    把一个窗口里每个快照的特征，再聚合成窗口级特征。
    这里用了四种聚合方式：均值、标准差、最后值、斜率。
"""
def aggregate_window_features(feature_rows: np.ndarray) -> np.ndarray:
    feature_rows = np.asarray(feature_rows, dtype=np.float64)
    if feature_rows.ndim != 2:
        raise ValueError('feature_rows must be 2-D')
    mean = feature_rows.mean(axis=0)
    std = feature_rows.std(axis=0)
    last = feature_rows[-1]
    first = feature_rows[0]
    slope = (last - first) / max(feature_rows.shape[0] - 1, 1)
    return np.concatenate([mean, std, last, slope], axis=0).astype(np.float32)

"""
    这是单个 acc_*.csv 快照的核心入口。
    输入一份快照的水平、垂直振动信号，输出一个特征字典。
"""
def extract_acc_features(horizontal: np.ndarray, vertical: np.ndarray, fs: float) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    feats.update(time_domain_features(horizontal, 'h'))
    feats.update(freq_domain_features(horizontal, fs, 'h'))
    feats.update(time_frequency_features(horizontal, 'h'))
    feats.update(time_domain_features(vertical, 'v'))
    feats.update(freq_domain_features(vertical, fs, 'v'))
    feats.update(time_frequency_features(vertical, 'v'))
    feats['hv_corr'] = float(np.corrcoef(horizontal, vertical)[0, 1]) if np.std(horizontal) > EPS and np.std(vertical) > EPS else 0.0
    feats['hv_energy_ratio'] = _safe_div(np.sum(np.square(horizontal)), np.sum(np.square(vertical)))
    feats['hv_rms_diff'] = float(np.sqrt(np.mean(np.square(horizontal))) - np.sqrt(np.mean(np.square(vertical))))
    return feats


def feature_columns_from_rows(rows: Iterable[Dict[str, float]]) -> List[str]:
    first = next(iter(rows))
    return list(first.keys())
