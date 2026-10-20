# DataReview

**EWAM-Tac / DataReviewer** — 面向机器人与触觉数据集的通用质量评分工具。

Schema-agnostic quality scorer for robotics / tactile datasets.  
通过 **Adapter + Role Mapping + Profile 门控** 适配异构格式与可扩展数据清单。

---

## 功能概览

| 能力 | 说明 |
|---|---|
| 多格式适配 | LeRobot v2.1、HDF5、文件夹 dumps（GelSight / DIGIT 等） |
| 角色映射 | 将各异 feature key 映射到 `state` / `action` / `tactile` / `force` / `hand_pose` / `image.*` |
| 七维评分 | completeness · integrity · dynamics · consistency · annotation · tactile · trainability |
| 目录画像 | 解析可扩展 CSV 清单（如 Notion 导出），按数据族门控维度 |
| 本地命中 | 在 `--search-root` 下模糊匹配并自动打开嵌套 `lerobot_v2.1` |

**Integrity 为硬门控**：该维 `fail` 时，整体 `status=fail`。

---

## 安装

```bash
git clone https://github.com/EWAM-Tac/DataReview.git
cd DataReview
pip install -r requirements.txt
```

依赖：`numpy`、`pyarrow`、`PyYAML`；可选 `opencv-python-headless`（视频探测）、`h5py`（HDF5）。

---

## 快速开始

### 单库评分

```bash
PYTHONPATH=$PWD python -m data_reviewer review \
  --dataset /path/to/lerobot_v2.1 \
  --max-episodes 20 \
  --out outputs/review.json
```

也支持传入样本树根目录（如 `SharpaDex-v1.0_sample10/`），会自动下钻找到 `meta/info.json`。

### 目录（Catalog）模式

CSV 是**可扩展参考清单**，不是写死的 55 条名单。追加行即可；评分逻辑不依赖具体数据集名。

```bash
PYTHONPATH=$PWD python -m data_reviewer catalog \
  --csv /path/to/Notion_Deduplicated_Datasets.csv \
  --search-root /path/to/local_data \
  --print-profiles \
  --out outputs/catalog_review.json
```

---

## 评分维度

| ID | 门控 | 说明 |
|---|---|---|
| completeness | 否 | 帧/模态覆盖 |
| integrity | **是** | NaN、时间戳、解码 |
| dynamics | 否 | 运动丰富度（action / state / hand_pose） |
| consistency | 否 | 跨模态一致性（action↔state、触觉同步） |
| annotation | 否 | 语言 / 子任务（无则 N/A） |
| tactile | 否 | taxel / deform / force-wrench 代理 |
| trainability | 否 | 划分、idle、异常值风险 |

缺失模态 → 该维 `N/A`，不参与加权平均。  
目录画像下，数据族未宣称的维度会被强制 `N/A`。

---

## 支持的数据格式

| Adapter | 探测条件 | 典型来源 |
|---|---|---|
| `LeRobotAdapter` | `meta/info.json` 或嵌套 `lerobot_v2.1/` | SharpaDex、T-Rex、多数 HF 机器人集 |
| `Hdf5Adapter` | `*.hdf5` / `*.h5` | FTP、旧 teleop、仿真 rollout |
| `FolderAdapter` | 图像/视频目录 + 可选 NPZ/CSV | GelSight/DIGIT、接触对 / 材料集 |

未知布局：扩展 `configs/default_roles.yaml`，或 `--roles` 传入自定义规则。

---

## 目录如何扩展

| 需求 | 做法 |
|---|---|
| 新增数据集 | CSV 追加一行（`数据集名称` + 可选模态/主体描述） |
| 新关键词/新族 | 编辑 `configs/profile_rules.yaml` |
| 难推断的行 | 可选列：`family`、`expects_tactile`、`expects_action` … |
| 额外列 | 写入 `profile.extra`，评分忽略但 JSON 保留 |
| 新磁盘布局 | 新增 `data_reviewer/adapters/` 或扩展 role YAML |

**数据族**（可覆盖）：`robot_vla` · `human_hoi` · `tactile_repr` · `static_grasp` · `mixed`

---

## 设计流程

1. **Catalog / Profile** — 从元数据文本推断模态与适用维度  
2. **Adapter** — 打开 LeRobot / HDF5 / folder → 统一 `EpisodeArrays`  
3. **Roles** — feature key → 语义角色  
4. **Scorers** — 独立打分，按 confidence 加权聚合；跳过 N/A

---

## 项目结构

```
DataReview/
├── README.md
├── requirements.txt
├── configs/
│   ├── default_roles.yaml      # 特征 → 角色正则
│   └── profile_rules.yaml      # 目录画像关键词 / 数据族规则
└── data_reviewer/
    ├── __main__.py             # CLI: review | catalog
    ├── adapters/               # LeRobot / HDF5 / Folder
    ├── metrics/                # 七维评分器
    ├── catalog.py              # 可扩展 CSV 加载
    ├── profile.py              # DatasetProfile + 推断
    └── roles.py                # role mapping
```

---

## License

Internal EWAM-Tac project. Contact maintainers for redistribution.
