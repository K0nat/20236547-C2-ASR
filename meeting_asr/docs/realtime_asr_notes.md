# 级联实时 ASR 技术说明

完整实现位于会议工程文件 `realtime/server.py`。该文件为组内共用入口，除 ASR 外还包含其他业务模块。

依据完整工程 README，级联路线可概括为：

```text
实时 ASR（Paraformer realtime）→ 上下文纠错与翻译 → 字幕输出
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

## 本说明不涉及的内容

以下内容由完整工程中其他模块实现，详细说明见该工程 README：

- 说话人相关实现：`online_vad.py`、`online_diarization.py` 及声纹匹配与会议级校正  
- 端到端实时翻译（LiveTranslate / Omni）  
- 会议纪要、RAG 检索、语音助手与 TTS  

本文档仅作为个人结题材料中对识别链路的索引说明，不能替代完整工程源码与系统说明。
