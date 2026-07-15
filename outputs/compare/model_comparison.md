# ASR Model Comparison

| Model | Size(MB) | Peak VRAM(MB) | Total(s) | Avg latency(s) | Avg WER | Avg CER | Scored |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 926.37 | 4353.55 | 137.69 | 0.0131 | 0.027256813680363756 | 0.016562807453310104 | 10242 |
| turbo | 1547.28 | 4443.18 | 467.76 | 0.0455 | 0.012855097125747393 | 0.007760107700854171 | 10242 |
| large_v3 | 2948.3 | 18778.71 | 600.14 | 0.0583 | 0.0135525055560205 | 0.007655108647211562 | 10242 |
| paraformer_en | 891.91 | 862.76 | 2488.84 | 0.2429 | 0.111267 | 0.100047 | 10242 |

**Best (lowest WER, then speed): `turbo`**

## Trade-off notes
- **small**: smallest / fastest baseline
- **turbo**: balanced speed vs accuracy
- **large_v3**: highest Whisper quality, more VRAM/time
- **paraformer_en**: non-autoregressive (FunASR), good for segment/long-audio pipeline
