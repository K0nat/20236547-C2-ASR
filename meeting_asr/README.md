# 会议系统 · C2 语音识别（个人提交仓）

> **负责人**：蔡润泽（20236547）  
> **说明**：本仓库仅归档本人在会议工程中与 **ASR 语音识别** 相关的代码与说明。完整会议系统（双语字幕、说话人、纪要、RAG、助手等）见团队工程与组长 README，**请勿将本仓理解为完整会议产品仓库**。

对齐组长工程 README 中的模块划分：

| 组长 README 中的 C2 描述 | 本仓库认领范围 |
|--------------------------|----------------|
| 级联实时 ASR（`paraformer-realtime-v2`） | ✅ 参与：PCM → Recognition → `asr_partial` / `asr_final` |
| 离线分段 ASR（`outputs/asr/asr_segments.json`） | ✅ 离线批处理脚本 |
| 说话人 / VAD 门控声纹 / CAM++ 等 | ❌ **不认领**（组长及说话人相关实现） |
| 纠错、翻译、端到端、纪要、RAG、TTS | ❌ **不认领** |

完整系统技术说明以团队仓库中的 `README.md` 为准。

---

## 1. 本人工作（尽量收敛）

1. **离线批处理 ASR**：对切片音频调用云端 ASR，写出 `asr_segments.json`。  
2. **级联实时 ASR 链路说明与关键回调**：实时入口位于完整工程 `realtime/server.py`（多人共用文件）。本仓用文档标明本人关注的识别回调与事件类型，**不单独拆分/重写整份 server**。

模型（与完整工程配置一致）：

- 离线：`qwen3-asr-flash`（见 `config/asr_related.snippet.yaml`）  
- 实时级联：`paraformer-realtime-v2`（完整工程环境变量 / `server.py` 默认值）

---

## 2. 本仓库文件

| 路径 | 说明 |
|------|------|
| `src/c2_asr_ali.py` | 离线批处理主脚本 `run()` |
| `src/utils/*.py` | 批处理依赖的共用工具（API 客户端等，**非本人独占模块**） |
| `config/asr_related.snippet.yaml` | 与 ASR 相关的配置摘录 |
| `.env.example` | 密钥占位（**勿提交真实 `.env`**） |
| `docs/realtime_asr_notes.md` | 实时识别在完整工程中的位置与函数说明 |

---

## 3. 实时部分在完整工程中的位置

完整工程路径示例：

`meeting_speech_system_ami/realtime/server.py`

与级联 ASR 直接相关的符号（详见 `docs/realtime_asr_notes.md`）：

- `QueueCallback` / `on_event` → `asr_partial`、`asr_final`
- `realtime_ws` 中创建 `Recognition`、`send_audio_frame`
- `start_asr`、`sender`（事件出队推前端）

同一文件中还包含声纹、翻译、纪要、助手等逻辑，**不属于本仓库认领范围**。

---

## 4. 运行说明

离线批处理需在**完整会议工程**目录下配置 `.env` 与 `config/config.yaml` 后运行（本仓为归档，不提供完整数据与密钥）：

```text
python -m src.c2_asr_ali
```

实时演示请按组长 README 启动后端 `realtime.server` 与 Streamlit 前端，选择 **级联** 模式。

---

## 5. 相关链接

- 实训评测 C2（Whisper 等）：https://github.com/K0nat/20236547-C2-ASR  
- 本部分路径：本仓库 `meeting_asr/` 目录；完整会议工程另见团队仓库。
