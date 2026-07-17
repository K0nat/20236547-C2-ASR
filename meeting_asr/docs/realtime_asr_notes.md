# 级联实时 ASR 技术说明

完整实现位于会议工程文件 `realtime/server.py`。本说明仅整理其中与 **实时语音识别** 相关的接口与数据流。

级联路线中的识别环节可概括为：

```text
PCM 音频帧 → Paraformer realtime → asr_partial / asr_final → 下游字幕链路
```

默认实时识别模型为 `paraformer-realtime-v2`。

## 与语音转写直接相关的接口

| 符号 | 作用 |
|------|------|
| `ASR_MODEL` / `Recognition(...)` | 级联模式下创建云端实时识别会话 |
| `QueueCallback.on_event` | 解析识别结果，写入 `asr_partial` 或 `asr_final` 事件 |
| `_extract_text` / `_is_final` | 提取文本，并判断是否为一句结束 |
| `start_asr` | 在独立线程中调用 `recognition.start()` |
| `recognition.send_audio_frame` | 发送 PCM 音频帧 |
| `sender` | 从事件队列取出结果并推送至前端 |

## 说明

本文档用于个人结题材料中对识别链路的索引说明；系统整体架构与其余模块说明见团队完整工程 README。
