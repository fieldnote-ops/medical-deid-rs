# 贡献与问题反馈

提交问题时说明程序版本、macOS 与芯片型号、运行命令、错误信息和预期行为。用虚构文本构造最小复现；不要提交真实病历、密钥、权重或本机环境目录。

修改后执行：

```bash
bash scripts/cargo.sh fmt --all -- --check
bash scripts/cargo.sh test --release --locked
bash scripts/cargo.sh clippy --release --all-targets --locked -- -D warnings
source scripts/env.sh
python3 -m unittest discover -s tests -v
python3 scripts/check_docs.py
```

涉及训练或模型打包时，请在新 run 目录跑训练和恢复，并从打包后安装的模型执行示例推理。不要覆盖已有检查点；报告使用的数据与指标口径。CI 检查源码和单元测试，首次大模型下载和完整训练需要本地验证。
