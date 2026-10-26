# 训练指南

先按[安装指南](installation.md)运行 setup.sh，准备固定版本的 hfl/rbt3、分词器和 Python 环境。训练与恢复由 Rust / Burn 执行。

## 跑通示例训练

```bash
bash scripts/setup.sh
bash scripts/train_demo.sh
```

生成 384 份训练、64 份 dev、96 份 test 虚构文书。默认训练 3 轮，batch size 8，学习率 0.0001，seed 42，最大 token 数 128，AdamW weight decay 0.01、epsilon 1e-8。默认使用 Metal。

demo-v0.1 目录已有训练状态时拒绝重开；再次实验请指定新目录：

```bash
source scripts/env.sh
bash scripts/train_demo.sh "$DEID_WORK/runs/demo-second"
```

## 使用自己的数据

将 train.jsonl、dev.jsonl、test.jsonl 按[数据格式](data-format.md)放到 DEID_WORK 下的数据目录。先检查，再训练：

```bash
source scripts/env.sh
bash scripts/deid.sh audit --dataset "$DEID_WORK/runs/my-data"
bash scripts/deid.sh train --model-dir "$DEID_WORK/models/rbt3" \
  --dataset "$DEID_WORK/runs/my-data" --run "$DEID_WORK/runs/my-model" \
  --epochs 3 --batch-size 8 --lr 0.0001 --seed 42 --max-tokens 128 --backend metal
```

人物组及完整文本在 train/dev 间不能重叠。训练每轮在 dev 上评测，best.bpk 按 dev 严格实体 F1 选优。test 留给最终评测，不能用于选检查点或调规则。训练超长样本会报错，不静默截断；请按任务规则提前分段并同步实体区间。

## 恢复训练

```bash
bash scripts/deid.sh train --model-dir "$DEID_WORK/models/rbt3" \
  --dataset "$DEID_WORK/runs/my-data" --run "$DEID_WORK/runs/my-model" \
  --resume --epochs 1 --batch-size 8 --lr 0.0001 --seed 42 --max-tokens 128 --backend metal
```

这里 epochs 表示追加轮数。恢复需要相同数据指纹、学习率、batch size、seed 和 max-tokens。请沿用相同后端。

| 产物 | 用途 |
|---|---|
| best.bpk | dev 选优的推理检查点 |
| checkpoints/epoch-NNNN | 每轮模型与优化器，用于恢复 |
| state.json | 当前轮数、步数、数据指纹和训练选项 |
| training-report.json | 梯度、验证指标和恢复检查摘要 |
| test-report.json | train_demo 完成后生成的留出集评测 |

训练结束后按[模型指南](models.md)打包推理检查点；自己的模型卡应说明数据来源和真实评测结果。相同 seed 不保证跨硬件、后端、版本逐字节得到相同权重。重新训练使用新的模型版本和文件哈希，无需匹配示例模型的校验值。
