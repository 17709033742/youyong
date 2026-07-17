# LoRA + Anchor Graph Spectral Clustering

最小可运行实验：LoRA（只调少量参数）提升特征质量 → Anchor Graph 谱聚类 → NMI/ARI/ACC 评估。

## 快速开始
```bash

pip install -r requirements.txt
python run_all.py          xx                   # 一键：微调(LoRA)→抽特征→聚类→评估