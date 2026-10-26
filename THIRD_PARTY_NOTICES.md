# 第三方来源与许可

项目原创代码采用 Apache-2.0。依赖、预训练模型和评测数据保留各自许可；本文件不是对全部依赖重新授权。Cargo.lock 和 requirements-model.txt 记录实际依赖版本。

[Rust 依赖来源索引](licenses/dependency-index.json)提供已解析依赖的版本、许可表达式、作者与上游地址，包含部分可选或仅用于构建的依赖；它不代表所有列出的组件都会进入可执行文件。

| 来源 | 用途 | 上游许可与说明 |
|---|---|---|
| [Burn](https://github.com/tracel-ai/burn) | 训练、张量与模型存储 | MIT OR Apache-2.0 |
| [CubeCL](https://github.com/tracel-ai/cubecl) | GPU 运行时与缓存配置 | MIT OR Apache-2.0 |
| [tracel-ai/models · bert-burn](https://github.com/tracel-ai/models/tree/238983f9bf739358a74a6cee676eb754fc74b5bf/bert-burn) | BERT 实现与权重加载 | MIT OR Apache-2.0；版权与 MIT 文本见 licenses/bert-burn-MIT.txt |
| [hfl/rbt3](https://huggingface.co/hfl/rbt3/tree/0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c) | 预训练模型、分词器与派生示例权重 | 模型卡标注 Apache-2.0；由 HFL 发布，来源和引用见下文 |
| [Meddies/meddies-pii-v2](https://huggingface.co/datasets/Meddies/meddies-pii-v2) | 模型卡中的公开合成评测证据 | CC-BY-NC-4.0；仅作本地非商业评测，未用于训练；不分发该数据 |

## HFL 模型引用

Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Ziqing Yang, Shijin Wang and Guoping Hu. *Revisiting Pre-Trained Models for Chinese Natural Language Processing*. Findings of EMNLP, 2020. [论文](https://aclanthology.org/2020.findings-emnlp.58/)。

基础模型源自 HFL 的三层 RoBERTa-wwm-ext。示例权重在本地微调，移除未使用的 pooler 和 MLM 头并添加 token 分类头。未暗示上游作者对本工具作出认可或质量保证。

训练自己的模型时，应保留实际基础模型、数据来源和相应限制。公开发布推理包前，请确认模型卡和清单描述的是该包的真实训练与评测情况。
