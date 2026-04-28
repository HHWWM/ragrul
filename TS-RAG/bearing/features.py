"""
把单个 CSV 里的水平/垂直振动，转成统计特征；
再把一个窗口里的多行特征，聚合成窗口级特征。

用途：
1. 为 HI 构造提供输入
2. 为 RAG 检索提供窗口级查询特征
3. 同时兼容时序建模和检索增强两部分
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pywt
from scipy.fft import rfft, rfftfreq

# ============================================================
# 一、数值稳定常量
# ============================================================

EPS = 1e-12

# ============================================================
# 二、你以后主要改这里：特征选择清单
# ============================================================
# 说明：
# 1. 每一项都是一个二元组：
#       (通道, 特征名)
#
# 2. 通道支持：
#       "h"  -> 水平振动
#       "v"  -> 垂直振动
#       "hv" -> 水平/垂直交叉特征
# ============================================================

SELECTED_FEATURES: List[Tuple[str, str]] = [
    # --------------------------------------------------------
    # A. 水平振动特征（你现在最常用）
    # --------------------------------------------------------
    ("h", "kurtosis"),           # 峭度
    ("h", "entropy"),            # 熵值
    ("h", "fractal_dimension"),  # 分形维数
    ("h", "peak_factor"),        # 峰值因子
    ("h", "pulse_factor"),       # 脉冲因子
    ("h", "crest_factor"),       # 裕度因子
    ("h", "energy_ratio"),       # 能量比值
    ("h", "spectral_flatness"),  # 谱线平坦度
    ("h", "mean"),               # 均值
    ("h", "variance"),           # 方差
    ("h", "skewness"),           # 偏度
    ("h", "peak_vibration"),     # 峰值振动
    ("h", "rms_vibration"),      # 有效值振动
    #("hv", "corr"),
    #("hv", "energy_ratio"),

    # --------------------------------------------------------
    # B. 垂直振动特征（默认先注释掉，需要时取消注释）
    # --------------------------------------------------------
    # ("v", "kurtosis"),
    # ("v", "entropy"),
    # ("v", "fractal_dimension"),
    # ("v", "peak_factor"),
    # ("v", "pulse_factor"),
    # ("v", "crest_factor"),
    # ("v", "energy_ratio"),
    # ("v", "spectral_flatness"),
    # ("v", "mean"),
    # ("v", "variance"),
    # ("v", "skewness"),
    # ("v", "peak_vibration"),
    # ("v", "rms_vibration"),

    # --------------------------------------------------------
    # C. hv 交叉特征（默认先注释掉，需要时取消注释）
    # --------------------------------------------------------
    # ("hv", "corr"),            # 水平/垂直相关系数
    # ("hv", "energy_ratio"),    # 水平能量 / 垂直能量
    # ("hv", "rms_diff"),        # 水平RMS - 垂直RMS
]

# ------------------------------------------------------------
# 可选特征总表（单通道）
#
# 时域：
# mean
# variance
# std
# rms_vibration
# peak_vibration
# p2p
# mean_abs
# skewness
# kurtosis
# peak_factor
# pulse_factor
# shape_factor
# crest_factor
# energy
# entropy
#
# 频域：
# spec_energy
# spec_entropy
# spectral_centroid
# spectral_bandwidth
# dominant_freq
# rolloff_85
# spectral_flatness
# band_0_20
# band_20_50
# band_50_80
# band_80_100
# energy_ratio
#
# 时频域：
# wavelet_entropy
# wavelet_total_energy
# wavelet_e0
# wavelet_e1
# wavelet_e2
# wavelet_e3
#
# 其它：
# fractal_dimension
#
# hv 交叉特征：
# corr
# energy_ratio
# rms_diff
# ------------------------------------------------------------

# ============================================================
# 三、窗口聚合配置
# ============================================================
# 这个聚合函数是给 retrieval 查询向量用的。
# 你也可以通过注释来控制窗口聚合的方式。
WINDOW_AGG_PARTS = [
    "mean",   # 窗口内均值
    "std",    # 窗口内标准差
    "last",   # 窗口最后一个快照的特征值
    "slope",  # 窗口首尾趋势
]

# ============================================================
# 四、基础工具函数
# ============================================================

def _to_1d_float(x: np.ndarray) -> np.ndarray:
    """
    把输入转成一维 float64，并去掉无效值。
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    return x


def _safe_div(a: float, b: float) -> float:
    """
    安全除法，避免分母过小。
    """
    return float(a / (b + EPS))


def _safe_entropy_from_values(values: np.ndarray) -> float:
    """
    安全计算熵，避免出现 log(0) 和除零。
    """
    values = np.asarray(values, dtype=np.float64)
    values = np.abs(values)

    total = float(values.sum())
    if total <= EPS:
        return 0.0

    p = values / total
    p = p[p > 0]
    return float(-(p * np.log(p + EPS)).sum())


def _power_spectrum(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算单边频谱和功率谱。
    """
    x = _to_1d_float(x)
    if len(x) < 4:
        return np.array([0.0], dtype=np.float64), np.array([0.0], dtype=np.float64)

    x = x - np.mean(x)
    spec = np.abs(rfft(x))
    power = np.square(spec)
    freqs = rfftfreq(len(x), d=1.0 / fs)

    return freqs, power

# ============================================================
# 五、单通道特征函数
# ============================================================

def feat_mean(x: np.ndarray, fs: float) -> float:
    x = _to_1d_float(x)
    return float(np.mean(x)) if len(x) > 0 else 0.0


def feat_variance(x: np.ndarray, fs: float) -> float:
    x = _to_1d_float(x)
    return float(np.var(x)) if len(x) > 0 else 0.0


def feat_std(x: np.ndarray, fs: float) -> float:
    x = _to_1d_float(x)
    return float(np.std(x)) if len(x) > 0 else 0.0


def feat_rms_vibration(x: np.ndarray, fs: float) -> float:
    x = _to_1d_float(x)
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x)) + EPS))


def feat_peak_vibration(x: np.ndarray, fs: float) -> float:
    x = _to_1d_float(x)
    if len(x) == 0:
        return 0.0
    return float(np.max(np.abs(x)))


def feat_p2p(x: np.ndarray, fs: float) -> float:
    x = _to_1d_float(x)
    if len(x) == 0:
        return 0.0
    return float(np.ptp(x))


def feat_mean_abs(x: np.ndarray, fs: float) -> float:
    x = _to_1d_float(x)
    if len(x) == 0:
        return 0.0
    return float(np.mean(np.abs(x)))


def feat_skewness(x: np.ndarray, fs: float) -> float:
    """
    偏度：三阶中心矩 / 标准差^3
    """
    x = _to_1d_float(x)
    if len(x) < 3:
        return 0.0

    mu = np.mean(x)
    std = np.std(x)
    if std <= EPS:
        return 0.0

    return float(np.mean(((x - mu) / std) ** 3))


def feat_kurtosis(x: np.ndarray, fs: float) -> float:
    """
    峭度：四阶中心矩 / 标准差^4
    注意：正态分布大约等于 3
    """
    x = _to_1d_float(x)
    if len(x) < 4:
        return 0.0

    mu = np.mean(x)
    std = np.std(x)
    if std <= EPS:
        return 0.0

    return float(np.mean(((x - mu) / std) ** 4))


def feat_peak_factor(x: np.ndarray, fs: float) -> float:
    """
    峰值因子 = peak / rms
    """
    peak = feat_peak_vibration(x, fs)
    rms = feat_rms_vibration(x, fs)
    return _safe_div(peak, rms)


def feat_pulse_factor(x: np.ndarray, fs: float) -> float:
    """
    脉冲因子 = peak / mean(abs(x))
    """
    peak = feat_peak_vibration(x, fs)
    mean_abs = feat_mean_abs(x, fs)
    return _safe_div(peak, mean_abs)


def feat_shape_factor(x: np.ndarray, fs: float) -> float:
    """
    波形因子 = rms / mean(abs(x))
    """
    rms = feat_rms_vibration(x, fs)
    mean_abs = feat_mean_abs(x, fs)
    return _safe_div(rms, mean_abs)


def feat_crest_factor(x: np.ndarray, fs: float) -> float:
    """
    裕度因子
    默认公式：
        peak / (mean(sqrt(abs(x))) ^ 2)

    注意：
    有些文献把“crest factor”写成 peak/rms，
    但你前面给的中文是“裕度因子”，
    更接近 clearance factor 的这个公式。
    """
    x = _to_1d_float(x)
    if len(x) == 0:
        return 0.0

    peak = feat_peak_vibration(x, fs)
    base = np.mean(np.sqrt(np.abs(x) + EPS))
    return _safe_div(peak, base * base)


def feat_energy(x: np.ndarray, fs: float) -> float:
    x = _to_1d_float(x)
    return float(np.sum(np.square(x))) if len(x) > 0 else 0.0


def feat_entropy(x: np.ndarray, fs: float) -> float:
    """
    幅值直方图熵
    """
    x = _to_1d_float(x)
    if len(x) < 4:
        return 0.0

    hist, _ = np.histogram(x, bins=64)
    return _safe_entropy_from_values(hist)


def feat_spec_energy(x: np.ndarray, fs: float) -> float:
    _, power = _power_spectrum(x, fs)
    return float(np.sum(power))


def feat_spec_entropy(x: np.ndarray, fs: float) -> float:
    _, power = _power_spectrum(x, fs)
    return _safe_entropy_from_values(power)


def feat_spectral_centroid(x: np.ndarray, fs: float) -> float:
    freqs, power = _power_spectrum(x, fs)
    power_sum = float(np.sum(power))
    if power_sum <= EPS:
        return 0.0
    return float(np.sum(freqs * power) / (power_sum + EPS))


def feat_spectral_bandwidth(x: np.ndarray, fs: float) -> float:
    freqs, power = _power_spectrum(x, fs)
    power_sum = float(np.sum(power))
    if power_sum <= EPS:
        return 0.0

    centroid = np.sum(freqs * power) / (power_sum + EPS)
    bw = np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / (power_sum + EPS))
    return float(bw)


def feat_dominant_freq(x: np.ndarray, fs: float) -> float:
    freqs, power = _power_spectrum(x, fs)
    if len(power) == 0:
        return 0.0
    idx = int(np.argmax(power))
    return float(freqs[idx])


def feat_rolloff_85(x: np.ndarray, fs: float) -> float:
    freqs, power = _power_spectrum(x, fs)
    power_sum = float(np.sum(power))
    if power_sum <= EPS:
        return 0.0

    cdf = np.cumsum(power) / (power_sum + EPS)
    idx = int(np.searchsorted(cdf, 0.85))
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def feat_spectral_flatness(x: np.ndarray, fs: float) -> float:
    """
    谱线平坦度 = 几何平均 / 算术平均
    """
    _, power = _power_spectrum(x, fs)

    if len(power) == 0:
        return 0.0

    power = power + EPS
    geo_mean = np.exp(np.mean(np.log(power)))
    arith_mean = np.mean(power)

    return _safe_div(geo_mean, arith_mean)


def feat_band_0_20(x: np.ndarray, fs: float) -> float:
    freqs, power = _power_spectrum(x, fs)
    total = float(np.sum(power))
    if total <= EPS:
        return 0.0

    nyq = fs / 2.0
    mask = (freqs >= 0.0) & (freqs < 0.2 * nyq)
    return float(np.sum(power[mask]) / (total + EPS))


def feat_band_20_50(x: np.ndarray, fs: float) -> float:
    freqs, power = _power_spectrum(x, fs)
    total = float(np.sum(power))
    if total <= EPS:
        return 0.0

    nyq = fs / 2.0
    mask = (freqs >= 0.2 * nyq) & (freqs < 0.5 * nyq)
    return float(np.sum(power[mask]) / (total + EPS))


def feat_band_50_80(x: np.ndarray, fs: float) -> float:
    freqs, power = _power_spectrum(x, fs)
    total = float(np.sum(power))
    if total <= EPS:
        return 0.0

    nyq = fs / 2.0
    mask = (freqs >= 0.5 * nyq) & (freqs < 0.8 * nyq)
    return float(np.sum(power[mask]) / (total + EPS))


def feat_band_80_100(x: np.ndarray, fs: float) -> float:
    freqs, power = _power_spectrum(x, fs)
    total = float(np.sum(power))
    if total <= EPS:
        return 0.0

    nyq = fs / 2.0
    mask = (freqs >= 0.8 * nyq) & (freqs <= 1.01 * nyq)
    return float(np.sum(power[mask]) / (total + EPS))


def feat_energy_ratio(x: np.ndarray, fs: float) -> float:
    """
    能量比值
    这里默认定义为：
        高频能量（50%~100% Nyquist） / 全频能量

    如果你论文里的 EnergyRatio 定义不同，
    只改这个函数即可。
    """
    freqs, power = _power_spectrum(x, fs)
    total = float(np.sum(power))
    if total <= EPS:
        return 0.0

    nyq = fs / 2.0
    mask = (freqs >= 0.5 * nyq) & (freqs <= 1.01 * nyq)
    high_energy = float(np.sum(power[mask]))
    return float(high_energy / (total + EPS))


def feat_wavelet_entropy(x: np.ndarray, fs: float) -> float:
    x = _to_1d_float(x)
    if len(x) < 8:
        return 0.0

    wavelet = "db4"
    level = min(3, pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len))
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)
    energies = np.array([np.sum(np.square(c)) for c in coeffs], dtype=np.float64)

    return _safe_entropy_from_values(energies)


def feat_wavelet_total_energy(x: np.ndarray, fs: float) -> float:
    x = _to_1d_float(x)
    if len(x) < 8:
        return 0.0

    wavelet = "db4"
    level = min(3, pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len))
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)
    energies = np.array([np.sum(np.square(c)) for c in coeffs], dtype=np.float64)

    return float(np.sum(energies))


def feat_wavelet_e0(x: np.ndarray, fs: float) -> float:
    return _wavelet_energy_ratio_by_index(x, idx=0)


def feat_wavelet_e1(x: np.ndarray, fs: float) -> float:
    return _wavelet_energy_ratio_by_index(x, idx=1)


def feat_wavelet_e2(x: np.ndarray, fs: float) -> float:
    return _wavelet_energy_ratio_by_index(x, idx=2)


def feat_wavelet_e3(x: np.ndarray, fs: float) -> float:
    return _wavelet_energy_ratio_by_index(x, idx=3)


def _wavelet_energy_ratio_by_index(x: np.ndarray, idx: int) -> float:
    """
    小波某个分量的能量占比
    """
    x = _to_1d_float(x)
    if len(x) < 8:
        return 0.0

    wavelet = "db4"
    level = min(3, pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len))
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)
    energies = np.array([np.sum(np.square(c)) for c in coeffs], dtype=np.float64)

    total = float(np.sum(energies))
    if total <= EPS:
        return 0.0

    if idx >= len(energies):
        return 0.0

    return float(energies[idx] / (total + EPS))


def feat_fractal_dimension(x: np.ndarray, fs: float) -> float:
    """
    分形维数
    这里采用 Katz Fractal Dimension
    """
    x = _to_1d_float(x)
    n = len(x)

    if n < 3:
        return 1.0

    diff = np.diff(x)
    L = np.sum(np.sqrt(1.0 + diff * diff))

    idx = np.arange(n, dtype=np.float64)
    d = np.max(np.sqrt((idx - 0.0) ** 2 + (x - x[0]) ** 2))

    if L <= EPS or d <= EPS:
        return 1.0

    val = np.log10(float(n)) / (np.log10(float(n)) + np.log10(d / (L + EPS) + EPS))
    return float(val)


# ============================================================
# 六、hv 交叉特征
# ============================================================

def feat_hv_corr(h: np.ndarray, v: np.ndarray, fs: float) -> float:
    """
    水平 / 垂直相关系数
    """
    h = _to_1d_float(h)
    v = _to_1d_float(v)

    n = min(len(h), len(v))
    if n < 2:
        return 0.0

    h = h[:n]
    v = v[:n]

    if np.std(h) <= EPS or np.std(v) <= EPS:
        return 0.0

    return float(np.corrcoef(h, v)[0, 1])


def feat_hv_energy_ratio(h: np.ndarray, v: np.ndarray, fs: float) -> float:
    """
    水平能量 / 垂直能量
    """
    h = _to_1d_float(h)
    v = _to_1d_float(v)

    e_h = float(np.sum(np.square(h)))
    e_v = float(np.sum(np.square(v)))

    return _safe_div(e_h, e_v)


def feat_hv_rms_diff(h: np.ndarray, v: np.ndarray, fs: float) -> float:
    """
    水平RMS - 垂直RMS
    """
    rms_h = feat_rms_vibration(h, fs)
    rms_v = feat_rms_vibration(v, fs)

    return float(rms_h - rms_v)


# ============================================================
# 七、特征注册表
# ============================================================

# 单通道特征：既可以给 h 用，也可以给 v 用
SINGLE_CHANNEL_FEATURES = {
    "mean": feat_mean,
    "variance": feat_variance,
    "std": feat_std,
    "rms_vibration": feat_rms_vibration,
    "peak_vibration": feat_peak_vibration,
    "p2p": feat_p2p,
    "mean_abs": feat_mean_abs,
    "skewness": feat_skewness,
    "kurtosis": feat_kurtosis,
    "peak_factor": feat_peak_factor,
    "pulse_factor": feat_pulse_factor,
    "shape_factor": feat_shape_factor,
    "crest_factor": feat_crest_factor,
    "energy": feat_energy,
    "entropy": feat_entropy,
    "spec_energy": feat_spec_energy,
    "spec_entropy": feat_spec_entropy,
    "spectral_centroid": feat_spectral_centroid,
    "spectral_bandwidth": feat_spectral_bandwidth,
    "dominant_freq": feat_dominant_freq,
    "rolloff_85": feat_rolloff_85,
    "spectral_flatness": feat_spectral_flatness,
    "band_0_20": feat_band_0_20,
    "band_20_50": feat_band_20_50,
    "band_50_80": feat_band_50_80,
    "band_80_100": feat_band_80_100,
    "energy_ratio": feat_energy_ratio,
    "wavelet_entropy": feat_wavelet_entropy,
    "wavelet_total_energy": feat_wavelet_total_energy,
    "wavelet_e0": feat_wavelet_e0,
    "wavelet_e1": feat_wavelet_e1,
    "wavelet_e2": feat_wavelet_e2,
    "wavelet_e3": feat_wavelet_e3,
    "fractal_dimension": feat_fractal_dimension,
}

# hv 交叉特征
CROSS_CHANNEL_FEATURES = {
    "corr": feat_hv_corr,
    "energy_ratio": feat_hv_energy_ratio,
    "rms_diff": feat_hv_rms_diff,
}


# ============================================================
# 八、主入口：单个 acc_*.csv 快照特征提取
# ============================================================

def extract_acc_features(horizontal: np.ndarray, vertical: np.ndarray, fs: float) -> Dict[str, float]:
    """
    这是你当前工程的核心入口函数。
    输入：
    - horizontal: 单个快照的水平振动
    - vertical:   单个快照的垂直振动
    - fs:         采样率

    输出：
    - 一个特征字典

    本函数的行为完全由顶部 SELECTED_FEATURES 决定。
    你以后只需要在那里注释/取消注释即可。
    """
    feats: Dict[str, float] = {}

    h = _to_1d_float(horizontal)
    v = _to_1d_float(vertical)

    for channel_name, feature_name in SELECTED_FEATURES:
        # -------------------------
        # 水平通道
        # -------------------------
        if channel_name == "h":
            if feature_name not in SINGLE_CHANNEL_FEATURES:
                raise KeyError(f"未知单通道特征: {feature_name}")

            val = SINGLE_CHANNEL_FEATURES[feature_name](h, fs)
            feats[f"h_{feature_name}"] = float(val)

        # -------------------------
        # 垂直通道
        # -------------------------
        elif channel_name == "v":
            if feature_name not in SINGLE_CHANNEL_FEATURES:
                raise KeyError(f"未知单通道特征: {feature_name}")

            val = SINGLE_CHANNEL_FEATURES[feature_name](v, fs)
            feats[f"v_{feature_name}"] = float(val)

        # -------------------------
        # hv 交叉特征
        # -------------------------
        elif channel_name == "hv":
            if feature_name not in CROSS_CHANNEL_FEATURES:
                raise KeyError(f"未知交叉特征: {feature_name}")

            val = CROSS_CHANNEL_FEATURES[feature_name](h, v, fs)
            feats[f"hv_{feature_name}"] = float(val)

        else:
            raise ValueError(f"未知通道类型: {channel_name}，只能是 'h' / 'v' / 'hv'")

    return feats


# ============================================================
# 九、窗口级聚合特征
# ============================================================

def _calc_slope(y: np.ndarray) -> float:
    """
    计算一段序列的线性趋势斜率。
    用于描述该特征在窗口内是上升还是下降。
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(y)

    if n < 2:
        return 0.0

    x = np.arange(n, dtype=np.float64)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    denom = np.sum((x - x_mean) ** 2)
    if denom <= EPS:
        return 0.0

    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    return float(slope)


def aggregate_window_features(feature_rows: np.ndarray) -> np.ndarray:
    """
    把一个窗口中每个快照的特征，再聚合成一个窗口级向量。

    输入：
    - feature_rows: [seq_len, num_features]

    输出：
    - 一维聚合向量，作为 retrieval 查询特征
    """
    feature_rows = np.asarray(feature_rows, dtype=np.float64)

    if feature_rows.ndim != 2:
        raise ValueError("feature_rows must be 2-D")

    parts = []

    if "mean" in WINDOW_AGG_PARTS:
        parts.append(feature_rows.mean(axis=0))

    if "std" in WINDOW_AGG_PARTS:
        parts.append(feature_rows.std(axis=0))

    if "last" in WINDOW_AGG_PARTS:
        parts.append(feature_rows[-1])

    if "slope" in WINDOW_AGG_PARTS:
        slope = np.array(
            [_calc_slope(feature_rows[:, i]) for i in range(feature_rows.shape[1])],
            dtype=np.float64,
        )
        parts.append(slope)

    if len(parts) == 0:
        # 万一你把上面全注释掉了，至少保底返回最后一行
        parts.append(feature_rows[-1])

    out = np.concatenate(parts, axis=0).astype(np.float32)
    return out


# ============================================================
# 十、特征列名工具
# ============================================================

def feature_columns_from_rows(rows: Iterable[Dict[str, float]]) -> List[str]:
    """
    从若干行特征字典里，拿到列名顺序。
    这里保留你原来的接口，兼容现有工程。
    """
    first = next(iter(rows))
    return list(first.keys())