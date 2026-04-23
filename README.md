# ragrul_tsrag_patch

这是按你的要求做的 **“基于 TS-RAG 原仓库增量改造”** 版本，不是整套重写。

## 这份代码怎么尽量保留了 TS-RAG 原实现

我把 TS-RAG 原仓库里最核心、最有辨识度的文件直接保留下来了：

- `TS-RAG/models/ChronosBolt.py`：原版 Chronos-Bolt + Retrieval ARM 实现，未重命名。
- `TS-RAG/models/base.py`
- `TS-RAG/models/moment.py`
- `TS-RAG/models/utils.py`
- `TS-RAG/dataset.py`
- `TS-RAG/pretrain.py`
- `TS-RAG/retrieve.py`
- `TS-RAG/zeroshot.py`

在这个基础上，我只新增了 `TS-RAG/bearing/*` 这套 **PHM2012 / acc-only / RUL** 适配层：

- `bearing/preprocess_phm2012.py`：读取 `Learning_set` / `Full_Test_Set`，只用 `acc_*.csv`
- `bearing/features.py`：时域 / 频域 / 时频域特征
- `bearing/retrieve_bearing.py`：按 TS-RAG `retrieve.py` 的风格构建 bearing RAG 数据库
- `bearing/dataset_bearing.py`：按 TS-RAG `dataset.py` 的风格构建训练/验证/测试集
- `bearing/models/chronosbolt_rul.py`：在原 `ChronosBoltModelForForecastingWithRetrieval` 上加一个最小 RUL 回归头
- `bearing/train_rul.py`：RUL 训练
- `bearing/evaluate_rul.py`：RUL 评估与可视化
- `bearing/infer_rul.py`：单次推理
- `bearing/thingsboard_mqtt.py`：把预测结果经 MQTT 发到 ThingsBoard

也就是说，这个版本的思路是：

**保留 TS-RAG 的骨架与 ARM 检索增强结构，只把“数据层”和“任务头”从 zero-shot forecasting 换成 PHM2012 的 bearing RUL regression。**

## 数据假设

- 数据源：`wkzs111/phm-ieee-2012-data-challenge-dataset`
- 只读取 `acc_*.csv`
- 默认使用加速度两通道：水平/垂直（CSV 第 5、6 列；0-based 为 4、5）
- 不使用温度
- `Learning_set` 内部按时间顺序切成 train/val
- `Full_Test_Set` 作为外部测试；如果你只有 `Test_set`，可以把 `splits_to_scan` 改成你自己要扫的目录

## 预处理与特征

每个 `acc_*.csv` 文件被当成一个“快照”，会提取：

- 时域：均值、方差、RMS、峰值、峰峰值、偏度、峭度、波形因子、脉冲因子、裕度因子等
- 频域：谱能量、谱熵、谱质心、带宽、主频、滚降频率、分段频带能量
- 时频域：小波分解能量与小波熵
- 双通道关系：相关系数、能量比、RMS 差

之后：

1. 用训练部分拟合 `StandardScaler`
2. 再用 PCA 把多维统计特征压成 1 个健康指标 `HI`
3. 用 `HI` 序列作为 Chronos-Bolt/TS-RAG 主干的输入
4. 用窗口聚合特征参与向量库嵌入拼接与检索

## RAG 数据库设计

数据库条目由训练窗口组成：

- `x`: 长度 `seq_len` 的 HI 序列
- `y_seq`: 后续 `prediction_length` 个时刻的归一化 RUL 曲线
- `y_rul_norm`: 当前窗口末端的归一化 RUL
- `query_features`: 当前窗口聚合后的多域统计特征
- `embedding`: `[TSFM窗口嵌入 ; feature embedding]` 拼接后的向量

检索时保留了 TS-RAG 的风格：

- 用 FAISS 建库
- Top-k 近邻检索
- 把检索到的 `(x, y_seq)` 拼回 `retrieved_seq`
- 再送进原 TS-RAG 的 ARM/MoE 融合逻辑

## 训练逻辑

`bearing/models/chronosbolt_rul.py` 没有推翻原 `ChronosBoltModelForForecastingWithRetrieval`，而是：

- 复用原 patching / encoder / decoder / ARM 融合流程
- 把最终输出头从 “quantile forecast head” 换成 “sigmoid RUL regression head”
- 损失改成 `SmoothL1Loss`

这比“整仓库重写”更贴近原作者风格。

## 建议超参数

默认在 `configs/phm2012_rul.yaml`：

- `seq_len: 64`
- `prediction_length: 16`
- `top_k: 5`
- `epochs: 40`
- `learning_rate: 5e-4`
- `freeze_backbone: true`

如果显存更大、数据更完整，可以试：

- `seq_len: 96`
- `top_k: 8`
- `epochs: 60`

## 评估指标

代码里输出两套指标：

- 归一化 RUL：MAE / RMSE / MAPE / R2 / PHM Score
- 以“剩余快照步数”计的 RUL：MAE / RMSE / MAPE / R2 / PHM Score

同时会保存每个 bearing 的预测曲线图。

## ThingsBoard

`bearing/thingsboard_mqtt.py` 会循环调用 `bearing/infer_rul.py`，然后发布：

- `pred_rul_norm`
- `topk_distance_mean`
- `bearing_dir`

到：

- topic: `v1/devices/me/telemetry`
- 用户名：`access_token`

你只要把 `configs/phm2012_rul.yaml` 里的 token/host/port 改成自己的即可。

## 使用步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备原 TS-RAG 基座权重

把原 TS-RAG 需要的 `Chronos-Bolt` 权重放到：

```bash
TS-RAG/checkpoints/base
```

至少要有：

```bash
autogluon_model.pth
config.json
```

### 3. 修改配置

编辑：

```bash
TS-RAG/configs/phm2012_rul.yaml
```

把：

- `dataset.root`
- `model.pretrained_model_path`
- `thingsboard.access_token`

改成你自己的路径/令牌。

### 4. 预处理 + 建库

```bash
cd TS-RAG
bash script/preprocess_phm2012_rul.sh
```

### 5. 训练

```bash
bash script/train_phm2012_rul.sh
```

### 6. 评估与可视化

```bash
bash script/eval_phm2012_rul.sh
```

### 7. 推送到 ThingsBoard


```bash
bash script/thingsboard_phm2012_rul.sh /path/to/one/bearing_dir
```

## 推到你自己的 GitHub 仓库

我这里不能直接替你向远端仓库写入，所以给你最短命令：

```bash
git clone https://github.com/HHWWM/ragrul.git
cd ragrul
cp -r /path/to/ragrul_tsrag_patch/* .
git add .
git commit -m "Adapt TS-RAG for PHM2012 bearing RUL with ThingsBoard MQTT"
git push origin main
```

## 你接下来最值得先改的 3 个地方

1. `configs/phm2012_rul.yaml` 里的路径与 token。
2. `bearing/preprocess_phm2012.py` 里的 split 方案：如果你要严格按 bearing 留出验证集，可以把当前“按时间切 train/val”改成“按 bearing 切”。
3. `bearing/infer_rul.py` 里的在线检索向量，目前为了单文件部署简单，做的是轻量检索；如果你要线上效果更接近离线训练，建议把线上 query embedding 改成和 `retrieve_bearing.py` 完全一致的 Chronos + feature 拼接流程。
