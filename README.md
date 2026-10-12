# Support Ticket Classification with LoRA

Fine-tuning Qwen2.5-0.5B-Instruct with LoRA for customer support
ticket classification.

## Tasks
- Predict a support queue
- Predict ticket priority
- Generate structured JSON predictions

## Model
- Base model: Qwen/Qwen2.5-0.5B-Instruct
- Fine-tuning: LoRA via PEFT
- Training examples: 18,998
- Validation examples: 2,375
- Training: 1 epoch, 1,188 steps

## Quick evaluation
The recorded quick evaluation used the first 100 validation examples:
- Queue accuracy: 43%
- Queue macro-F1: 0.3202
- Priority accuracy: 39%
- Priority macro-F1: 0.2957
- Valid JSON rate: 100%

These are preliminary results, not a full-validation benchmark.
The TF-IDF baseline has not yet been re-evaluated on this exact same
100-example subset, so direct comparison is not established.

## Adapter usage
Install compatible versions of transformers, peft, and torch.
Load the base model `Qwen/Qwen2.5-0.5B-Instruct`, then load this
repository's adapter using `PeftModel.from_pretrained`.

## Files
- `adapter_model.safetensors`: trained LoRA adapter
- `adapter_config.json`: adapter configuration
- `tokenizer.json` and tokenizer configuration files
- `quick_evaluation.csv`: per-example predictions
- `evaluation_summary.json`: summary metrics
- `training_metrics.json`: training metrics
- `training_history.json`: training log history

## Limitations
Results are preliminary. Evaluate on the complete held-out validation
set and compare against baselines using identical examples before
making performance claims.
