# 会议系统 · C2 语音识别模块说明

**负责人**：蔡润泽（学号：20236547）  
**目录位置**：个人代码库 `meeting_asr/`  
**对应完整工程**：组内「会议双语字幕与会议知识助手」仓库（以该仓库 README 为系统总说明）

本目录用于提交本人在会议工程中与 **自动语音识别（ASR）** 相关的代码与技术说明，并非完整会议系统源码包。系统中的说话人区分、上下文纠错与翻译、端到端语音翻译、会议纪要、RAG 检索及 TTS 等功能，由组内其他成员负责实现，详见完整工程文档。

按完整工程 README 的模块划分，与本目录相关的对应关系如下：

| 完整工程中的 C2 相关能力 | 本目录覆盖内容 |
|--------------------------|----------------|
| 级联实时 ASR（`paraformer-realtime-v2`） | 实时识别链路说明：PCM 输入 → Recognition → `asr_partial` / `asr_final` |
| 离线分段 ASR（`outputs/asr/asr_segments.json`） | 离线批处理脚本及依赖说明 |
| 说话人识别（VAD 门控、声纹嵌入与聚类等） | 由组长负责，本目录不予展开 |
| 纠错、翻译、端到端、纪要、RAG、TTS | 由其他成员负责，本目录不予展开 |

---

## 1. 模块工作内容

1. **离线批处理 ASR**：按音频切片调用云端 ASR 接口，生成并保存 `asr_segments.json`。  
2. **级联实时 ASR**：实时服务入口位于完整工程文件 `realtime/server.py`（组内共用）。本目录以文档形式说明其中与识别相关的回调与事件类型；不复制该文件中与识别无关的业务逻辑。

**所用模型（与完整工程配置一致）：**

- 离线识别：`qwen3-asr-flash`（参见 `config/asr_related.snippet.yaml`）  
- 实时级联识别：`paraformer-realtime-v2`（完整工程环境变量或 `server.py` 中的默认配置）

---

## 2. 目录结构

| 路径 | 说明 |
|------|------|
| `src/c2_asr_ali.py` | 离线批处理主脚本（入口函数 `run()`） |
| `src/utils/*.py` | 批处理所需的工程共用工具（如 API 客户端等） |
| `config/asr_related.snippet.yaml` | 与 ASR 相关的配置字段摘录 |
| `.env.example` | 环境变量示例（请勿将含真实密钥的 `.env` 提交至仓库） |
| `docs/realtime_asr_notes.md` | 实时识别相关符号与在完整工程中的位置说明 |

---

## 3. 实时识别在完整工程中的位置

完整工程参考路径：

`meeting_speech_system_ami/realtime/server.py`

与级联 ASR 直接相关的主要符号（详见 `docs/realtime_asr_notes.md`）：

- `QueueCallback` / `on_event`：产生 `asr_partial`、`asr_final` 事件  
- `realtime_ws`：级联模式下创建 `Recognition`，并调用 `send_audio_frame`  
- `start_asr`、`sender`：启动识别会话，并将事件推送至前端  

该文件同时包含说话人、翻译、纪要与助手等逻辑，分属其他模块，本文仅说明识别相关部分。

---

## 4. 运行说明

离线批处理需在完整会议工程环境中配置 `.env` 与 `config/config.yaml` 后执行：

```text
python -m src.c2_asr_ali
```

实时演示请按照完整工程 README 启动后端 `realtime.server` 与 Streamlit 前端，并选择 **级联** 模式。

---

## 5. 相关链接

- 实训评测 C2（Whisper 批量识别与实验）：https://github.com/K0nat/20236547-C2-ASR  
- 会议 ASR 说明目录：本仓库下的 `meeting_asr/`  
- 完整会议工程：见团队合并代码库及其 README  
