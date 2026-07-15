# 级联实时 ASR 说明（相对完整工程）

完整实现位于团队会议工程：`realtime/server.py`（多人共用，含 ASR 以外大量逻辑）。

根据组长 README，级联路线为：

```text
实时 ASR（Paraformer realtime）→（他人）纠错/翻译 → 字幕
```

默认实时识别模型：`paraformer-realtime-v2`。

## 与「语音→文本」直接相关的符号

| 符号 | 作用 |
|------|------|
| `ASR_MODEL` / `Recognition(...)` | 级联模式下创建云端实时识别 |
| `QueueCallback.on_event` | 回调解析文本，入队 `asr_partial` 或 `asr_final` |
| `_extract_text` / `_is_final` | 从 SDK 结果取文本、判断句末 |
| `start_asr` | 线程中 `recognition.start()` |
| `recognition.send_audio_frame` | 发送 PCM |
| `sender` | 从队列取出事件推送到前端 |

## 明确不在本文档范围

- `online_vad.py`、`online_diarization.py`、声纹标签与会议级说话人校正（组长 README「说话人识别」）
- LiveTranslate / Omni 端到端路线
- 纪要、RAG、语音助手、TTS

本文件仅作个人结题材料中的索引说明，不替代完整工程源码。
