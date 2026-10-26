# 数据格式与标签

训练、dev 和带标注评测文件均为 UTF-8 JSONL，一行一份文书。样例见 [training-example.jsonl](../examples/training-example.jsonl)。

| 字段 | 含义 |
|---|---|
| id | 文件内唯一文书 ID |
| group | 人物或患者组；同一人物的所有文书必须进入同一划分 |
| template | 文书或模板类型，用于数据管理 |
| text | 原始文本 |
| spans | 按 start 升序排列、不相交的实体区间；无实体时为 [] |

span 包含 `start`、`end`、`kind`。位置是 UTF-8 **字节偏移**，end 不包含在区间内。例如「患者林清禾」中的姓名从字节 6 开始，到字节 15 结束。中文字符、emoji 和英文字母的字节长度不同。区间不得切开字符或分词 token；程序检测到不一致时停止。

| kind | 标注范围 |
|---|---|
| PERSON | 患者、家属及签署者姓名 |
| ADDRESS | 应被遮盖的地址 |
| PHONE | 电话号码 |
| EMAIL | 邮箱 |
| ID | 证件、病案及其他约定标识号 |
| DATE | 具体日期 |
| HOSPITAL | 医院或当前任务约定的机构名称 |

HOSPITAL 的范围应在自己的标注规范中统一，不能在不同划分中混用医院和任意机构的口径。年龄、职业、私人链接、口令及间接识别信息不在当前模型标签中。

目录中使用 train.jsonl、dev.jsonl 和 test.jsonl。audit 检查两两划分的人物组及完整文本重叠，并校验实体区间。它不替代语义近重复检查、授权审查或人工标注复核。

批量推理输入仅需 id、text，见[使用指南](usage.md)。
