# Remaining Optional Downloads

The core GPU stack is already installed and verified:

- `torch==2.6.0+cu124`
- `torchvision==0.21.0+cu124`
- `torchaudio==2.6.0+cu124`
- `vllm==0.8.5.post1`
- `xformers==0.0.29.post2`
- `transformers==4.51.3`
- `ray==2.55.1`
- `flashinfer-python==0.2.2.post1+cu124torch2.6`

## FlashInfer

FlashInfer is installed and pinned to `0.2.2.post1+cu124torch2.6`. This is intentional for `vllm==0.8.5.post1`: newer FlashInfer builds are detected as not backward-compatible by this vLLM version and get disabled.

The verified vLLM smoke log is `logs/vllm_flashinfer_022_smoke.log`; it includes:

`Using FlashInfer for top-p & top-k sampling.`

## FlashAttention Python Package

The standalone `flash-attn` package is not installed. A prebuilt `flash-attn==2.8.3.post1` wheel imported with an ABI error against the local `torch==2.6.0+cu124`, and a source build was intentionally stopped because it is too CPU/RAM-heavy for this server.

Current training defaults avoid requiring it:

- Hugging Face actor/SFT model loading uses `attn_implementation=sdpa`.
- `use_remove_padding=False`, because verl's padding-removal path imports `flash_attn.bert_padding`.
- vLLM rollout still uses its own Flash Attention backend and FlashInfer sampling, so rollout acceleration is active.

If a later multi-GPU image has more build headroom, install `flash-attn` in that image and then enable `USE_REMOVE_PADDING=True`.
