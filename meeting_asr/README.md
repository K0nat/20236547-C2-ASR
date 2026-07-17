# 会议系统 · C2 实时流式 ASR 模块说明

**负责人**：蔡润泽（学号：20236547）  
**目录位置**：个人代码库 `meeting_asr/`  
**对应完整工程**：组内会议/课堂双语字幕与端到端语音翻译演示系统  

本目录收录本人在会议工程中与 **自动语音识别（ASR）** 相关的实现与技术说明，作为个人模块交付材料，完整系统说明以团队工程仓库为准。

本模块职责范围如下：

1. **实训侧**：C2 ASR 语音识别模型推理与评测（详见仓库根目录 README）  
2. **会议侧**：级联路线下的 **实时流式 ASR**，以及离线批处理 ASR 脚本归档  

| 本模块能力 | 说明 |
|------------|------|
| 级联实时流式 ASR（`paraformer-realtime-v2`） | PCM 输入 → Recognition → `asr_partial` / `asr_final` |
| 离线批处理 ASR（`qwen3-asr-flash`） | 分段识别并输出 `asr_segments.json` |

---

## 1. 模块工作内容

1. **实时流式 ASR**：完整工程入口为 `realtime/server.py`。本目录说明其中与识别相关的回调机制与事件类型。  
2. **离线批处理 ASR**：按音频切片调用云端 ASR 接口，生成结构化转写结果。

**模型配置（与完整工程一致）：**

- 实时级联识别：`paraformer-realtime-v2`  
- 离线识别：`qwen3-asr-flash`（参见 `config/asr_related.snippet.yaml`）

---

## 2. 目录结构

| 路径 | 说明 |
|------|------|
| `src/c2_asr_ali.py` | 离线批处理主脚本（入口函数 `run()`） |
| `src/utils/*.py` | 批处理所需工程共用工具 |
| `config/asr_related.snippet.yaml` | ASR 相关配置字段摘录 |
| `.env.example` | 环境变量示例（勿提交含真实密钥的 `.env`） |
| `docs/realtime_asr_notes.md` | 实时识别相关符号及在完整工程中的位置说明 |

---

## 3. 实时识别在完整工程中的位置

完整工程参考路径：`meeting_speech_system_ami/realtime/server.py`

与级联 ASR 直接相关的主要符号（详见 `docs/realtime_asr_notes.md`）：

- `QueueCallback` / `on_event`：产生 `asr_partial`、`asr_final`  
- `realtime_ws`：级联模式下创建 `Recognition`，并调用 `send_audio_frame`  
- `start_asr`、`sender`：启动识别会话，并将识别事件推送至前端  

---

## 4. 运行说明

离线批处理需在完整会议工程环境中配置 `.env` 与 `config/config.yaml` 后执行：

```text
python -m src.c2_asr_ali
```

实时演示请按完整工程 README 启动后端 `realtime.server` 与前端，并选择 **级联** 模式。

---

## 5. 相关链接

- 实训评测 C2（Whisper 批量识别与实验）：https://github.com/K0nat/20236547-C2-ASR  
- 会议 ASR 说明目录：本仓库 `meeting_asr/`  
- 团队合并代码库：https://github.com/GPAUP/nlp_team  
