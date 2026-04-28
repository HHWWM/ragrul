from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # 服务器环境下保存图片，不弹窗

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# PHM2012 的 acc_*.csv 中：
# 第 5 列（0-based=4）是水平振动
# 第 6 列（0-based=5）是垂直振动
ACC_COLS = (4, 5)


def read_yaml(path: Path) -> Dict:
    """
    读取 yaml 配置文件。
    """
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_acc_file(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取单个 acc_*.csv 文件，返回：
    - h_sig: 水平振动信号
    - v_sig: 垂直振动信号

    这里做了比较稳健的解析：
    1. 尝试常见分隔符
    2. 自动去掉全空列
    3. 自动把非法值转成 NaN 后再过滤
    """
    sep_list = [",", ";", "\t", r"\s+"]
    last_shape = None

    for sep in sep_list:
        try:
            df = pd.read_csv(
                csv_path,
                header=None,
                sep=sep,
                engine="python" if sep in [";", "\t", r"\s+"] else None,
            )

            # 去掉全空列
            df = df.dropna(axis=1, how="all")
            last_shape = df.shape

            # 至少要有 6 列，才能读到第 5、6 列
            if df.shape[1] < 6:
                continue

            h_sig = pd.to_numeric(df.iloc[:, ACC_COLS[0]], errors="coerce").to_numpy(dtype=np.float32)
            v_sig = pd.to_numeric(df.iloc[:, ACC_COLS[1]], errors="coerce").to_numpy(dtype=np.float32)

            ok_mask = np.isfinite(h_sig) & np.isfinite(v_sig)
            h_sig = h_sig[ok_mask]
            v_sig = v_sig[ok_mask]

            if len(h_sig) == 0 or len(v_sig) == 0:
                continue

            return h_sig, v_sig

        except Exception:
            continue

    raise ValueError(f"无法解析文件: {csv_path}, 最近一次 shape={last_shape}")


def iter_bearing_dirs(data_root: Path, split_name: str) -> List[Path]:
    """
    扫描某个 split（如 Learning_set / Full_Test_Set）下面的所有轴承目录。
    """
    split_dir = data_root / split_name
    if not split_dir.exists():
        return []

    return sorted([p for p in split_dir.iterdir() if p.is_dir()])


def collect_full_life_signal(bearing_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """
    读取某个轴承目录下全部 acc_*.csv，
    按文件名排序后，顺序拼接成“完整全寿命信号”。

    返回：
    - h_all: 完整水平振动
    - v_all: 完整垂直振动
    - cut_pos: 每个快照结束后的全局边界位置，用于画分界线
    - file_names: 成功读取的文件名列表
    """
    file_list = sorted(
        [
            p for p in bearing_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".csv" and p.name.startswith("acc_")
        ]
    )

    if not file_list:
        raise FileNotFoundError(f"{bearing_dir} 下没有找到 acc_*.csv 文件")

    h_parts: List[np.ndarray] = []
    v_parts: List[np.ndarray] = []
    cut_pos: List[int] = []
    file_names: List[str] = []

    total_len = 0

    for csv_path in file_list:
        try:
            h_sig, v_sig = load_acc_file(csv_path)
        except Exception as e:
            print(f"[WARN] 跳过坏文件: {csv_path} -> {e}")
            continue

        h_parts.append(h_sig)
        v_parts.append(v_sig)
        total_len += len(h_sig)
        cut_pos.append(total_len)
        file_names.append(csv_path.name)

    if not h_parts:
        raise RuntimeError(f"{bearing_dir} 中所有 acc_*.csv 都读取失败")

    h_all = np.concatenate(h_parts, axis=0)
    v_all = np.concatenate(v_parts, axis=0)

    return h_all, v_all, cut_pos, file_names


def plot_one_bearing(
    bearing_id: str,
    split_name: str,
    h_all: np.ndarray,
    v_all: np.ndarray,
    save_dir: Path,
    fs_hz: float = 25600.0,
    draw_boundary: bool = False,
    max_boundary: int = 80,
) -> Path:
    """
    为单个轴承画“完整全寿命”的水平/垂直振动图。

    设计说明：
    1. 不截断，不只画单个 csv，而是画完整拼接后的全寿命信号
    2. 一张图里放两个子图：
       - 上：完整水平振动
       - 下：完整垂直振动
    3. 横轴默认用“全局采样点索引”，更稳定也更直观
       如果你更想看时间，也可以换成秒
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # 使用“全局采样点索引”作为横轴
    x_idx = np.arange(len(h_all), dtype=np.int64)

    # 也可以用秒做横轴，这里先保留，便于后续改
    x_sec = x_idx / float(fs_hz)

    fig, axes = plt.subplots(2, 1, figsize=(24, 10), dpi=150, sharex=True)

    # ===== 上图：水平振动 =====
    axes[0].plot(x_sec, h_all, linewidth=0.25)
    axes[0].set_title(f"{bearing_id} 完整全寿命水平振动信号", fontsize=14)
    axes[0].set_ylabel("幅值", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # ===== 下图：垂直振动 =====
    axes[1].plot(x_sec, v_all, linewidth=0.25)
    axes[1].set_title(f"{bearing_id} 完整全寿命垂直振动信号", fontsize=14)
    axes[1].set_ylabel("幅值", fontsize=12)
    axes[1].set_xlabel("时间（秒）", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.3)

    # 总标题：把关键信息写清楚
    fig.suptitle(
        f"轴承 {bearing_id} | 数据集: {split_name} | 完整全寿命信号 | 总采样点数: {len(h_all):,}",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = save_dir / f"{split_name}_{bearing_id}_full_life_signal.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return save_path


def save_one_bearing_plot(
    bearing_dir: Path,
    split_name: str,
    save_dir: Path,
    fs_hz: float,
) -> Dict:
    """
    处理单个轴承：
    1. 收集完整全寿命水平/垂直信号
    2. 生成图片
    3. 返回该轴承的元信息，后面可以汇总成 json
    """
    bearing_id = bearing_dir.name

    h_all, v_all, cut_pos, file_names = collect_full_life_signal(bearing_dir)

    img_path = plot_one_bearing(
        bearing_id=bearing_id,
        split_name=split_name,
        h_all=h_all,
        v_all=v_all,
        save_dir=save_dir,
        fs_hz=fs_hz,
        draw_boundary=False,   # 你目前要求的是完整图，不强调单个 csv 边界
        max_boundary=80,
    )

    info = {
        "split_name": split_name,
        "bearing_id": bearing_id,
        "num_csv_files": len(file_names),
        "num_samples_h": int(len(h_all)),
        "num_samples_v": int(len(v_all)),
        "image_path": str(img_path),
    }
    return info


def batch_plot_full_life_signals(
    data_root: Path,
    split_list: List[str],
    save_dir: Path,
    fs_hz: float = 25600.0,
) -> List[Dict]:
    """
    批量为多个 split 下的全部轴承生成完整全寿命振动图。
    """
    all_info: List[Dict] = []

    for split_name in split_list:
        bearing_dirs = iter_bearing_dirs(data_root, split_name)

        if not bearing_dirs:
            print(f"[WARN] split 不存在或为空: {split_name}")
            continue

        for bearing_dir in bearing_dirs:
            try:
                info = save_one_bearing_plot(
                    bearing_dir=bearing_dir,
                    split_name=split_name,
                    save_dir=save_dir,
                    fs_hz=fs_hz,
                )
                all_info.append(info)
                print(f"[OK] 已生成: {info['image_path']}")
            except Exception as e:
                print(f"[WARN] 生成失败: {bearing_dir} -> {e}")

    return all_info


def main() -> None:
    """
    命令行入口。
    用法示例：
    python -m bearing.visualize_full_life_signals --config configs/phm2012_rul.yaml
    """
    parser = argparse.ArgumentParser(description="为每个轴承生成完整全寿命水平/垂直振动信号图")
    parser.add_argument("--config", type=str, required=True, help="yaml 配置文件路径")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="图片保存目录；不填则默认保存到 processed_dir/full_life_signal_plots",
    )
    args = parser.parse_args()

    cfg = read_yaml(Path(args.config))
    data_root = Path(cfg["dataset"]["root"]).expanduser().resolve()
    fs_hz = float(cfg["dataset"].get("sampling_rate_hz", 25600))
    split_list = cfg["dataset"].get("splits_to_scan", ["Learning_set", "Full_Test_Set"])

    if args.save_dir:
        save_dir = Path(args.save_dir).expanduser().resolve()
    else:
        save_dir = Path(cfg["dataset"]["processed_dir"]).expanduser().resolve() / "full_life_signal_plots"

    save_dir.mkdir(parents=True, exist_ok=True)

    all_info = batch_plot_full_life_signals(
        data_root=data_root,
        split_list=split_list,
        save_dir=save_dir,
        fs_hz=fs_hz,
    )

    # 保存一个汇总 json，后续写论文、做答辩、查结果都方便
    report_path = save_dir / "full_life_signal_plot_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(all_info, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "save_dir": str(save_dir),
        "num_bearings": len(all_info),
        "report_path": str(report_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()