# C2 ASR 错误样例分析

> **负责人**：蔡润泽（20236547）  
> **模块**：C2 ASR 语音识别  
> **数据来源**：`outputs/eval/turbo/asr_predictions.json`  
> **模型**：whisper-large-v3-turbo  
> **评测规模**：10242 条（TESS + CREMA-D）  
> **整体指标**：平均 WER ≈ 0.0129；WER > 0 的样本 451 条（约 4.4%）

---

## 1. 文档目的

1. 展示 C2 在哪些场景下容易出现识别错误  
2. 为 C3 级联翻译提供「ASR 错误 → 翻译偏差」分析素材  
3. 支撑全组讨论：级联系统中错误是否在 ASR 阶段就已产生  

---

## 2. 错误类型说明

| 错误类型 | 含义 | 对 C3 的影响 |
|----------|------|-------------|
| 替换 / 语义偏离 | 识别结果与参考句语义不一致 | 翻译内容完全偏离原意 |
| 插入 / 幻觉 | 模型输出参考句中未出现的内容 | 译文可能多出问句、重复或无关信息 |
| 漏词 / 截断 | 仅识别出少量词或整句丢失 | C3 输入信息不足，无法正确翻译 |
| 情感 / 发音干扰 | CREMA 情感语音发音变形 | 短句也易被误识，错误率高于 TESS 朗读句 |

---

## 3. 典型错误样例（5 条）

### Case 1 — 强情感语音 + 整句语义偏离

| 字段 | 内容 |
|------|------|
| **id** | `cremad_1063_ieo_sad_md` |
| **数据集** | CREMA-D（悲伤 sad） |
| **reference** | It's 11 o'clock. |
| **prediction** | I have to learn the Clark. |
| **WER** | 2.000 |
| **错误类型** | 替换 / 语义偏离 |
| **可能原因** | 悲伤情感下发音变形严重，模型未对齐到原句，而是生成了语义无关的句子 |
| **对 C3 影响** | C3 会翻译「learn the Clark」，与「11 点」完全无关，级联错误根因在 ASR |

---

### Case 2 — 模型插入未说内容（幻觉）

| 字段 | 内容 |
|------|------|
| **id** | `cremad_1047_ieo_dis_md` |
| **数据集** | CREMA-D（厌恶 dis） |
| **reference** | It's 11 o'clock. |
| **prediction** | What time is it? It's 11 o'clock. |
| **WER** | 1.333 |
| **错误类型** | 插入 / 幻觉 |
| **可能原因** | 模型补全了常见「问时间」模板，在短句前插入了 `What time is it?` |
| **对 C3 影响** | 译文可能变成「几点了？11 点了」这类冗余问答，而非单纯报时 |

---

### Case 3 — 不同句型下的整句替换

| 字段 | 内容 |
|------|------|
| **id** | `cremad_1056_tai_ang_xx` |
| **数据集** | CREMA-D（愤怒 ang） |
| **reference** | The airplane is almost full. |
| **prediction** | Maybe tomorrow it will be cold. |
| **WER** | 1.200 |
| **错误类型** | 替换 / 语义偏离 |
| **可能原因** | 情感化发音 + 语言模型先验，导致输出与音频无关的另一完整句子 |
| **对 C3 影响** | 「飞机快满了」会被译成「明天可能很冷」，下游翻译无法纠正 ASR 错误 |

---

### Case 4 — 严重漏词 / 截断

| 字段 | 内容 |
|------|------|
| **id** | `cremad_1076_mti_sad_xx` |
| **数据集** | CREMA-D（悲伤 sad） |
| **reference** | Maybe tomorrow it will be cold. |
| **prediction** | you |
| **WER** | 1.000 |
| **错误类型** | 漏词 / 截断 |
| **可能原因** | 情感弱音、句首模糊，模型只捕捉到片段 `you`，其余内容丢失 |
| **对 C3 影响** | C3 仅收到单词 `you`，无法恢复完整语义，翻译必然失败 |

---

### Case 5 — 中等 WER 的语义替换

| 字段 | 内容 |
|------|------|
| **id** | `cremad_1045_iwl_dis_xx` |
| **数据集** | CREMA-D（厌恶 dis） |
| **reference** | I would like a new alarm clock. |
| **prediction** | I think I have a doctor's appointment. |
| **WER** | 0.857 |
| **错误类型** | 替换 / 语义偏离 |
| **可能原因** | 句长相近、语法结构类似，模型用另一常见句式替代原句 |
| **对 C3 影响** | 「想要新闹钟」变成「有医生预约」，翻译主题完全改变 |

---

## 4. 小结

1. **turbo 全量 WER 很低（≈1.3%）**，但错误主要集中在 **CREMA-D 情感语音**，TESS 朗读句相对稳定。  
2. 典型错误以 **语义偏离、插入幻觉、截断漏词** 为主，说明级联链路中 C3 的输入质量高度依赖 C2。  
3. 验收与联调时，C3 应固定读取 `outputs/latest/asr_predictions.json`；对高 WER 样本可结合本文件分析「听错导致译错」。  
4. 本文件 5 条样例均来自 turbo 全量结果中 WER 最高的候选集，具有代表性。

---

## 5. 附录：筛选命令

```bash
cd /root/siton-tmp/multimodal/c2/code
python <<'PY'
import json
rows = json.load(open("../outputs/eval/turbo/asr_predictions.json"))
errs = [r for r in rows if (r.get("wer") or 0) > 0]
errs.sort(key=lambda x: x["wer"], reverse=True)
for r in errs[:15]:
    print(r["id"], r["wer"], r["reference_text"], "->", r["prediction_text"])
PY