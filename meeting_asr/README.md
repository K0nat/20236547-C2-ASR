# 会议系统 · C2 实时流式 ASR 模块说明

**负责人**：蔡润泽（学号：20236547）  
**目录位置**：个人代码库 `meeting_asr/`  
**对应完整工程**：组内「会议/课堂双语字幕与端到端语音翻译」演示系统（以该仓库 README 为系统总说明）

本目录归档本人在会议工程中与 **自动语音识别（ASR）** 相关的代码与技术说明，并非完整会议系统源码包。

按组内最终分工，本人负责：

1. **C2 ASR 语音识别模型推理**（实训评测见仓库根目录 README）  
2. **会议系统中的实时流式 ASR**

说话人标签、会议记录与纪要、混合 RAG、级联纠错与翻译、端到端路线、TTS 与助手协议等，由组长及其他成员负责。

| 能力 | 本目录覆盖 |
|------|------------|
| 级联实时流式 ASR（`paraformer-realtime-v2`） | ✅ PCM → Recognition → `asr_partial` / `asr_final` |
| 离线批处理 ASR（`qwen3-asr-flash` → `asr_segments.json`） | ✅ 脚本与配置摘录 |
| 说话人识别 / 声纹 | ❌ 组长负责 |
| 纠错、翻译、端到端、纪要、RAG、TTS | ❌ 其他成员负责 |

---

## 1. 模块工作内容

1. **实时流式 ASR**：完整工程入口 `realtime/server.py`（组内共用）。本目录说明其中与识别相关的回调与事件类型。  
2. **离线批处理 ASR**：按音频切片调用云端 ASR，生成 `asr_segments.json`。

**所用模型（与完整工程配置一致）：**

- 实时级联识别：`paraformer-realtime-v2`  
- 离线识别：`qwen3-asr-flash`（参见 `config/asr_related.snippet.yaml`）

---

## 2. 目录结构

| 路径 | 说明 |
|------|------|
| `src/c2_asr_ali.py` | 离线批处理主脚本（入口函数 `run()`） |
| `src/utils/*.py` | 批处理所需的工程共用工具 |
| `config/asr_related.snippet.yaml` | 与 ASR 相关的配置字段摘录 |
| `.env.example` | 环境变量示例（请勿提交含真实密钥的 `.env`） |
| `docs/realtime_asr_notes.md` | 实时识别相关符号与在完整工程中的位置说明 |

---

## 3. 实时识别在完整工程中的位置

完整工程参考路径：`meeting_speech_system_ami/realtime/server.py`

与级联 ASR 直接相关的主要符号（详见 `docs/realtime_asr_notes.md`）：

- `QueueCallback` / `on_event`：产生 `asr_partial`、`asr_final`  
- `realtime_ws`：级联模式下创建 `Recognition`，并调用 `send_audio_frame`  
- `start_asr`、`sender`：启动识别会话，并将事件推送至前端  

该文件同时包含说话人、翻译、纪要与助手等逻辑，分属其他模块。

---

## 4. 运行说明

离线批处理需在完整会议工程环境中配置 `.env` 与 `config/config.yaml` 后执行：

```text
python -m src.c2_asr_ali
```

实时演示请按照完整工程 README 启动后端 `realtime.server` 与前端，并选择 **级联** 模式。

---

## 5. 相关链接

- 实训评测 C2（Whisper 批量识别与实验）：https://github.com/K0nat/20236547-C2-ASR  
- 会议 ASR 说明目录：本仓库下的 `meeting_asr/`  
- 完整会议工程：见团队合并代码库及其 README  
