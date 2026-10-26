# 模型安装与分发

规则脱敏不需要模型。联合脱敏与批量模型推理需要配套的 Burn 推理包。

## 安装推理包

从模型发布页获取 tar.gz 和可信的 SHA-256 值。推荐模型文件单独托管在 ModelScope，源码仓库保存版本清单；首次公开上传完成后再使用实际下载链接。当前源码不假定远端模型已经发布。[参考模型清单](../model-manifests/rbt3-synthetic-demo-v0.1.json)记录配套文件及压缩包校验值。

```bash
source scripts/env.sh
DEID_MODEL_SHA256=$(cut -d ' ' -f1 "$DEID_WORK/dist/demo-model.tar.gz.sha256")
bash scripts/model.sh install \
  --archive "$DEID_WORK/dist/demo-model.tar.gz" \
  --sha256 "$DEID_MODEL_SHA256" \
  --output "$DEID_WORK/models/demo"
bash scripts/model.sh verify --directory "$DEID_WORK/models/demo"
```

命令检查整个压缩包的哈希、内部文件清单、各文件大小和哈希、标签顺序，拒绝路径穿越、链接和重复文件。哈希应来自可信的发布渠道；同一不可信来源提供文件和哈希不能证明来源真实。目标目录已存在时停止。

推理包包含 best.bpk、burn-config.json、tokenizer.json、labels.json、manifest.json、模型卡及许可声明。不包含训练数据、优化器或虚拟环境。best.bpk 使用 Burn 格式，需对应的 v0.1 运行器。

## 打包训练结果

完成[训练](training.md)并填写自己的模型卡后：

```bash
source scripts/env.sh
bash scripts/model.sh pack \
  --checkpoint "$DEID_WORK/runs/demo-v0.1/best.bpk" \
  --model-dir "$DEID_WORK/models/rbt3" \
  --model-card MODEL_CARD.md \
  --model-id rbt3-synthetic-demo-v0.1 \
  --output "$DEID_WORK/dist/demo-model.tar.gz"
```

旁边生成 demo-model.tar.gz.sha256。默认模型卡对应示例模型；自有数据训练的模型应指定自己的卡片，说明用途、数据范围、参数及评测。当前打包器面向本项目 Apache-2.0 示例模型；使用自有数据或其他基础模型分发权重前，应另行确认相应许可并调整声明。

## 示例模型

`rbt3-synthetic-demo-v0.1` 在 hfl/rbt3 上使用虚构文书训练，仅用于演示训练与推理。能力和质量局限见[模型卡](../MODEL_CARD.md)。不建议把示例模型的文件哈希作为重新训练成功的判据；下载已有版本需要核对哈希，新训练则生成新的清单。
