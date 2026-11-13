# OBM-LM

**Ouroboros Brain Market — Language Model**

在零成本、极端受限的算力环境（8 GB 内存 CPU 沙箱 + 免费超算队列）下，验证「细胞市场经济」这一神经网络组织范式的可行性与可扩展性。

> 核心赌注：**多域不遗忘 + 分工深度 + 路由自知**，三样机制会随规模复利。

## 技术文档

完整技术报告见 [docs/technical-report.md](docs/technical-report.md)，涵盖架构设计、训练结果、方法演进、语料工程、事故复盘、前沿对比与路线图。

## 关键结果速览

| 指标 | 结果 |
| --- | --- |
| 模型规模 | 10.6 M → **176.8 M**（D=512 / L=8 / B=16 / 16k vocab / ctx 128） |
| 世代 | scale1~11，四域（en_lit / zh_classic / code_py / en_wiki） |
| 各代四域 ppl | scale8 113 / 800 / **55** / 277 · scale9 112 / **656** / 62 / 233 · scale10 127 / 761 / 66 / **201** · scale11 域内最优 **86 / 565 / 51** / 200 |
| 生成质量 | 全程峰值 top-1 token 命中率 0.195 → **0.2215**（scale11 全史新高） |
| 训练量 | 67,800 步，146.9 M tokens，tok/param 1.94（按 75 M 基线计） |
| 机制验证 | NLL 路由 top-1 决策与逐细胞实测最优 **100% 一致** |
| 市场运转 | scale11 实录 113 次封存、4 次分裂、3 次凋亡；7 细胞自组织 |
| 算力效率 | DCU 首战：万步 8.6 分钟，CPU → DCU 提速 **65×**（步时 2400 → 28 ms） |
| 部署栈 | 市场训练 + NLL 路由 + 跨代 zoo + 温度配方；零额外训练成本 |

## 仓库结构

```
.
├── docs/
│   ├── technical-report.md          # 完整技术报告
│   └── fast_architecture.md         # 快速架构说明
├── code/                            # 可运行代码栈
│   ├── obm/                         # 细胞市场核心（cell/mesh/router/stream）
│   ├── obm_lm/                      # 语言模型层（autograd/cell_lm/corpus/tokenizer）
│   ├── experiments/                 # 训练主控、实验、演示、战报脚本
│   ├── tools/                       # scnet 平台客户端 / 夜间流水线
│   ├── diag/                        # 诊断脚本集
│   └── train*.sh                    # scale4~9 训练脚本
├── reports/                         # 16 份战报 + 设计文档 + 训练方法
└── results/                         # 历史成绩单、scale8~11 日志与训练结果
```

模型权重（scale4~11）、四域 token 缓存与 torch 离线 wheel 体积较大，随 [Releases](https://github.com/gpt-dot/OBM-LM/releases) 发布：

| 归档 | 大小 | 内容 |
| --- | ---: | --- |
| `OBM-LM-scale9-posttrain.zip` | 257 MB | scale9 训后权重（101.04 M / 4 细胞），MD5 `1d9bd29d…` |
| `OBM-LM-scale10-posttrain.zip` | 353 MB | scale10 训后权重（151.56 M / 6 细胞），MD5 `78f4a094…` |
| `OBM-LM-scale11-posttrain.zip` | 401 MB | scale11 训后权重（176.82 M / 7 细胞），MD5 `773fdc86…` |
| `OBM-LM-scale8-posttrain-75M.zip` | 209 MB | scale8 训后权重（75.78 M / 3 细胞），MD5 `5deefd69…` |
| `OBM-LM-permanent-20261101.zip` | 836 MB | scale4~7 权重 + 12 份战报 + 69 份成绩单 + 设计文档 |
| `OBM-LM-toolkit-20261101.zip` | 347 MB | 完整代码 + tokenizer/四域缓存 + torch 离线 wheel |

## 快速开始

```bash
git clone https://github.com/gpt-dot/OBM-LM.git
cd OBM-LM

# 1) 下载并解压权重 / 缓存（Release 资产）
#    scale8 权重 zip 解压后得到 OBM-LM-scale8-训后-75M.pkl
#    将 OBM-LM-scale*.pkl 放入 models/ 或 code/models/

# 2) 启动 zoo 推理演示（端口 8765）
python3 code/experiments/demo_server.py --port 8765

# 3) 远端训练（需自行设置平台环境变量）
export SCNET_USER=... SCNET_AK=... SCNET_SK=...
export SCNET_REGION=... SCNET_JOBMANAGER_ID=... SCNET_HPC_BASE=...
export SCNET_WORK_DIR=... REMOTE_HOME=...
python3 code/tools/scnet_submit.py --script code/train9.sh --name scale9
```

训练脚本中的 `REMOTE_HOME` 默认值为 `/public/home/your_account`，运行前请覆盖为实际路径。

## 复现说明

- `code/experiments/obm_lab.py` 是训练主控，`run_scale.py` / `run_scale_fast.py` 为规模实验入口；
- `code/experiments/gen_archive.py` 生成跨代固定 prompt 归档；
- `code/experiments/territory_probe.py` 为领地探针；
- `code/tools/scnet_client.py` 通过 `SCNET_USER` / `SCNET_AK` / `SCNET_SK` 读取凭据，不含任何硬编码密钥；
- 所有脚本均支持 `OBM_ROOT` / `OBM_ZOO` / `REMOTE_HOME` / `OBM_WORKBASE` 环境变量覆盖路径。

详细数据口径与已知限制见技术报告 [附录 B](docs/technical-report.md#附录-b--数据口径与说明)。

## 安全与脱敏

仓库及 Release 资产中的超算账号、AK/SK、手机号、远端家目录、区域编号、API 地址等敏感信息均已移除或改为环境变量。若发现遗漏，请提 Issue。

## License

代码采用 [MIT License](LICENSE)。模型权重、实验报告与成绩单为研究产物，同样按 MIT 条件提供；第三方数据集与 wheel 遵循其原始许可。
