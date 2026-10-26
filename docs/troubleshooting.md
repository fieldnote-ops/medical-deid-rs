# 常见问题

**提示没有配置存储路径**：复制 .deid.env.example 为 .deid.env，编辑 DEID_STORAGE_ROOT。请在项目根目录通过 scripts 中的包装脚本运行。

**提示外置盘未挂载**：路径应指向已挂载的卷根目录。连接并挂载卷后重试。程序不会自动创建同名系统盘目录作为替代。

**提示找不到可执行文件**：源码版先执行 `bash scripts/cargo.sh build --release --locked`。预编译包检查 bin/medical-deid 是否完整及平台是否匹配。

**下载失败**：首次 setup 需要网络，默认优先 HF Mirror。可在 .deid.env 中设置其他 HF_ENDPOINT；程序只向入口请求公开预训练模型，不发送输入文书。

**模型包哈希错误**：重新获取文件和发布页校验值，不要绕过检查。模型安装目录已存在时，请为新版本使用新目录。

**JSONL 区间错误**：检查 UTF-8 字节偏移、end 不含末字符、排序、不重叠，以及分词 token 边界。[数据格式](data-format.md)有说明。

**恢复失败**：恢复需要保留 checkpoints 与 state.json，并沿用原数据和超参数。只有 best.bpk 的推理包不能恢复优化器。

**模型漏姓名或误删检验值**：查阅[模型卡](../MODEL_CARD.md)，在自己的独立留出集评测并人工复核输出。当前示例模型的覆盖范围有限。
