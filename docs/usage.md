# 使用指南

所有命令从项目根目录运行。先完成[安装](installation.md)。

## 单文件规则脱敏

```bash
bash scripts/deid.sh redact --input examples/example.txt
```

返回 JSON，例如输入「电话13800000000」时，结果中对应文本变成「电话[PHONE]」。实体位置引用**原始输入**，以 UTF-8 字节计数，end 不包含在区间内。

```json
{"schema_version":1,"redacted_text":"电话[PHONE]","spans":[{"start":6,"end":17,"kind":"PHONE"}],"offset_unit":"utf8_bytes_end_exclusive","review_required":true}
```

`review_required` 表示输出仍需检查。未被识别的隐私会留在输出中；该标记不表示已经完成复核。

## 模型与规则联合脱敏

安装模型到 models/demo 后运行：

```bash
source scripts/env.sh
bash scripts/deid.sh redact --input examples/example.txt \
  --model-dir "$DEID_WORK/models/demo" \
  --checkpoint "$DEID_WORK/models/demo/best.bpk" --backend metal
```

`--model-dir` 与 `--checkpoint` 必须成对提供。省略它们时使用规则。`--backend cpu` 可切换为 CPU；首次 Metal 推理可能包含编译耗时。长文本按重叠窗口处理。

## JSONL 批量推理

每行只需 `id` 和 `text`，id 在文件内唯一。示例见 [batch.jsonl](../examples/batch.jsonl)。不需要标注；即使附带标注字段，推理也会忽略它们。

```bash
source scripts/env.sh
bash scripts/deid.sh predict --dataset examples/batch.jsonl \
  --model-dir "$DEID_WORK/models/demo" \
  --checkpoint "$DEID_WORK/models/demo/best.bpk" \
  --output "$DEID_WORK/runs/batch-predictions.jsonl"
```

每份结果包含 id、`redacted_text`、模型 spans、`rules_spans` 和用于替换的 `combined_spans`。输出不含原始实体值列表。旁边的 `.metadata.json` 记录模型参数量、输入指纹和推理时间，不含原文。

所有命名的输出路径必须为 DEID_WORK 内的绝对路径。输出可包含尚未识别的敏感文本，应按原始资料的访问范围保存。当前命令会覆盖同名预测和评测文件，保留历史时请使用新文件名。

## 命令帮助

```bash
bash scripts/deid.sh --help
bash scripts/deid.sh train --help
bash scripts/deid.sh predict --help
```
