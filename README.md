A streamlined implementation for fine-tuning Meta's LLaMA 3.1 8B Instruct model using LoRA (Low-Rank Adaptation) with the PEFT library. This script enables efficient parameter-efficient fine-tuning on custom conversational datasets.

### Features
- **LoRA Configuration**: Low-rank adaptation targeting attention projection layers (q_proj, k_proj, v_proj, o_proj)
- **Memory Efficient**: Uses bfloat16 precision and gradient accumulation for training on limited GPU memory
- **Chat Template Support**: Automatically formats conversational data using the model's chat template
- **Checkpointing**: Saves model checkpoints every 50 steps with TensorBoard logging
- **Single GPU Optimized**: Configured for cuda:0 with pinned device mapping

### Training Configuration
- LoRA rank: 8, alpha: 16, dropout: 0.05
- Batch size: 2 per device with 8 gradient accumulation steps
- Learning rate: 4e-5 over 7 epochs
- Max sequence length: 4096 tokens

### Requirements
- PyTorch with CUDA support
- Transformers, PEFT, Datasets libraries
- JSONL dataset with `messages` field in chat format

### Output
Trained LoRA adapter saved to `./results_lora/final_adapter/` along with tokenizer and training arguments.
