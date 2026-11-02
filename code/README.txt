OBM-LM 运行工具包（2026-11-01）
====================================
与「永久产物包」互补：本包是【要用的/有用的】——解压即可运行。
所有脚本已去掉硬编码绝对路径：解压后按脚本自身位置自动定位包根
（_ROOT = 脚本上两级目录），权重按文件名模式自动搜索匹配
（models/ -> 包根 -> 上级目录 -> /workspace）。

目录结构
  code/
    obm/           细胞市场核心（cell/mesh/router/stream/baselines）
    obm_lm/        语言模型层（autograd/cell_lm/corpus/pure_bpe/tokenizer...）
    experiments/   obm_lab.py 训练主控、run_scale.py、demo_sandbox.py、
                   gen_archive.py 跨代生成档案、demo_server.py Flask 演示、
                   territory_probe.py 领地探针、model_soup.py、report_* 等
    tools/         scnet_client/submit/upload（算力平台）、auto_chain.py、
                   night_pipeline.py（夜间流水线）、upload_corpus.py
    diag/          diag_*.py 诊断脚本集、smoke.py、test_autograd.py
    data/cache/    tokenizer.json + 四域 token 缓存（约 160MB）
    train*.sh      scale5~9 远端作业脚本（REMOTE_HOME 环境变量可覆盖远端家目录）
  wheels/          torch 1.13.1+cpu whl（远端无外网时解压安装用）

快速上手
  1) 推理（zoo 模式，权重放 code/models/ 或任意候选目录）：
     cd code/experiments
     python3 demo_server.py --port 8765        # 自动搜索 scale5+scale6 权重
     # 或 OBM_ZOO="/path/a.pkl,/path/b.pkl" 覆盖
  2) 训练（远端）：train*.sh 内路径用 ${REMOTE_HOME:-默认} 解析，
     提交：python3 code/tools/scnet_submit.py --script code/train9.sh ...
  3) 数据缓存重建（若换语料）：data/encode_cache2.py 逐块追加写盘防 OOM。

环境变量
  OBM_ROOT      覆盖包根定位（默认自动推导）
  OBM_ZOO       覆盖 zoo 权重列表
  REMOTE_HOME   覆盖远端家目录（默认 /public/home/your_account）
  SCNET_USER / SCNET_AK / SCNET_SK
                超算平台凭据（必须自行设置，代码不含任何密钥）
  SCNET_REGION / SCNET_JOBMANAGER_ID / SCNET_HPC_BASE / SCNET_WORK_DIR
                平台区域、作业管理器、API 基址与工作目录
  OBM_WORKBASE  night_pipeline/auto_chain 工作产物目录（默认包根）
