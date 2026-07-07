# Teaching Source Policy

The rendered curriculum stores sources per chapter in `curriculum.js` using:

```js
{ label, url, type: 'official' | 'university' | 'paper' | 'supplemental', note }
```

V1 source map prioritizes official, university, and paper references:

- PyTorch official tutorials for tensor, autograd, module, DataLoader, and SDPA concepts.
- Hugging Face documentation for generation/KV cache and PEFT LoRA.
- Stanford CS224N/CS231N for implementation-heavy NLP and vision pedagogy.
- ViT and CLIP papers for vision/multimodal foundations.
- ExecuTorch and ONNX Runtime documentation for on-device/export/quantization topics.
- OpenAI Structured Outputs and LangGraph documentation for application-layer bridges.

The app renders per-chapter source links and a full source map in the final checklist chapter.
