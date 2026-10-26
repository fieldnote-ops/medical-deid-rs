# 安装与存储配置

支持范围：Apple Silicon macOS；开发验证使用 Apple M1 Max。Metal 加速和 CPU 推理均已在本机验证。其他系统尚未列入支持范围。

## 从源码构建

安装 Git、Rust 1.99.1、Python 3.13、uv 和 Xcode Command Line Tools。仓库的 rust-toolchain.toml 固定工具链，Cargo.lock 固定 Rust 依赖；requirements-model.txt 固定模型准备所需 Python 包。

在克隆后的仓库根目录执行：

```bash
cp .deid.env.example .deid.env
```

编辑 `.deid.env`，将 `DEID_STORAGE_ROOT` 设为外置卷的**挂载根目录**，保持 `DEID_REQUIRE_EXTERNAL=1`。例如外置卷名为 YourSSD 时，使用 `/Volumes/YourSSD`。该配置文件被 Git 忽略。

```bash
bash scripts/cargo.sh build --release --locked
bash scripts/deid.sh --version
bash scripts/deid.sh redact --input examples/example.txt
```

规则脱敏不需要 Python 模型环境。训练或准备预训练模型时运行：

```bash
bash scripts/setup.sh
```

预训练权重优先从 HF Mirror 获取，失败后尝试上游；下载固定 revision 并核对权重 SHA-256。首次准备需要网络，训练与推理使用本地模型。

## 预编译运行包

下载与平台对应的 `medical-deid-rs-0.1.0-macos-arm64.tar.gz` 和校验文件，核对 SHA-256 后解压。在解压目录中复制、编辑 `.deid.env.example`，然后通过 `bash scripts/deid.sh` 运行。运行包只包含推理程序和使用文档，模型包需另外安装。训练需要源码版。

## 文件存放位置

`DEID_WORK` 自动设为 `$DEID_STORAGE_ROOT/medical-deid-rs`。

| 子目录 | 内容 |
|---|---|
| target、cargo-home | Rust 编译与依赖缓存 |
| venv、python、cache | Python 环境、运行时与包缓存 |
| cache/cubecl | GPU 编译与调优缓存 |
| models | 预训练模型和已安装推理包 |
| runs | 数据、训练状态、预测和评测输出 |
| dist、tmp | 分发包和临时文件 |

外置盘未挂载时启动失败，不会自动改用系统盘。自动化环境可以显式设置 `DEID_REQUIRE_EXTERNAL=0`，仍需配置绝对存储路径；这用于没有外置卷的 CI。请通过包装脚本运行 Cargo 和程序。
