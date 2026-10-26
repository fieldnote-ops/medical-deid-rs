# Medical De-ID · Rust

本地中文医疗文本脱敏、轻量模型微调与质量评测工具。适合学习、开发和离线验证文本脱敏流程。

- 使用规则识别电话、邮箱、日期及部分证件、病案标识。
- 使用 Rust / Burn 微调 RBT3，补充人名、地址和机构等实体识别。
- 支持单文件脱敏、JSONL 批量推理、断点续训与逐类评测。
- 模型推理读取本地文件；编译、模型、缓存和 Python 环境统一使用配置的存储卷。

**v0.1.0 为实验版 CLI。** 已验证的平台为 Apple Silicon macOS，支持 Metal 和 CPU。当前模型仍会漏脱敏和误删，输出要求人工复核；不包含匿名化保证、医院接入或自动放行流程。

## 开始使用

需要 Rust 1.99.1、Python 3.13、uv 和 Xcode Command Line Tools。仅使用预编译运行包时，无需 Rust、uv 和 Xcode。

在项目根目录配置存储卷：

```bash
cp .deid.env.example .deid.env
# 编辑 .deid.env，把 DEID_STORAGE_ROOT 改为已挂载的外置盘路径。
bash scripts/cargo.sh build --release --locked
bash scripts/deid.sh redact --input examples/example.txt
```

无需权重即可运行上述规则脱敏。输出为 JSON，包含 `redacted_text`、实体字节区间和 `review_required: true`；肌酐和血压示例保留。完整输入输出见[使用指南](docs/usage.md)。

使用模型时，先按[模型安装指南](docs/models.md)安装推理包，再运行：

```bash
source scripts/env.sh
bash scripts/deid.sh redact --input examples/example.txt \
  --model-dir "$DEID_WORK/models/demo" \
  --checkpoint "$DEID_WORK/models/demo/best.bpk"
```

## 按任务查阅

| 任务 | 文档 |
|---|---|
| 安装、平台和存储配置 | [安装指南](docs/installation.md) |
| 单文件脱敏、批量推理、输出格式 | [使用指南](docs/usage.md) |
| 下载后安装、自训练模型打包 | [模型安装与分发](docs/models.md) |
| 准备数据、微调、恢复训练 | [训练指南](docs/training.md) |
| 实体标签与 JSONL 数据格式 | [数据格式](docs/data-format.md) |
| 在自己的留出集上评测 | [评测指南](docs/evaluation.md) |
| 模型能力与已知限制 | [模型卡](MODEL_CARD.md) |
| 安装或运行失败 | [常见问题](docs/troubleshooting.md) |

## 训练示例

```bash
bash scripts/setup.sh
bash scripts/train_demo.sh
```

示例使用程序生成的虚构数据，按人物组划分训练、验证和测试集。自己的数据需要按文档准备并检查授权范围。合成样例上的成绩不能替代实际使用场景的验证。

## 许可与反馈

项目代码采用 [Apache-2.0](LICENSE)。上游模型、依赖和评测数据各自的来源见[第三方声明](THIRD_PARTY_NOTICES.md)。问题报告和改进建议见[贡献指南](CONTRIBUTING.md)；提交问题时使用虚构最小样例，避免附带真实病历。
