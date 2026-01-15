# any_nn

A lightweight PyTorch training utility library featuring EMA (Exponential Moving Average) support and a flexible trainer class built on top of Hugging Face Accelerate.

## Features

- **AnyEMA**: Easy-to-use Exponential Moving Average implementation for model parameters
- **AnyTrainer**: Flexible training loop with built-in support for:
  - Mixed precision training (fp16/bf16)
  - Gradient accumulation
  - Checkpoint saving and resuming
  - TensorBoard logging
  - Evaluation during training
  - Multi-GPU training via Accelerate

## Installation

### From Source (pip)

Clone the repository and install in development mode:

```bash
git clone https://github.com/forhaxed/any_nn.git
cd any_nn
pip install -e .
```

Or install directly from the repository:

```bash
pip install git+https://github.com/forhaxed/any_nn.git
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- accelerate
- colorama
- tqdm

## Quick Start

### Using AnyEMA

```python
import torch
from any_nn import AnyEMA

# Create a model
model = torch.nn.Linear(10, 10)

# Initialize EMA
ema = AnyEMA(model.named_parameters())

# During training, update EMA after each step
for batch in dataloader:
    # ... training step ...
    ema.update(model.named_parameters(), decay=0.999)

# For evaluation, swap EMA parameters with model parameters
ema.swap(model.named_parameters())
# ... evaluate ...
ema.swap(model.named_parameters())  # Swap back
```

### Using AnyTrainer

```python
import torch
from any_nn import AnyTrainer

class MyTrainer(AnyTrainer):
    def __init__(self, output_dir="./output"):
        super().__init__(output_dir)
    
    def train_step(self, step, batch, device, weight_dtype):
        inputs, targets = batch
        inputs = inputs.to(device, dtype=weight_dtype)
        targets = targets.to(device)
        
        outputs = self.models[0](inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        
        return loss, {"mse_loss": loss}

# Setup trainer
trainer = MyTrainer(output_dir="./output")
trainer.models = [model]
trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
trainer.train_dataloader = train_dataloader
trainer.batch_size = 32
trainer.epochs = 10
trainer.gradient_accumulation_steps = 1
trainer.save_checkpoint_every_steps = 100
trainer.eval_every_steps = 50
trainer.mixed_precision = "bf16"  # or "fp16", "no"

# Initialize and train
trainer.init()
trainer.train()
```

### Resuming from Checkpoint

```python
trainer = MyTrainer(output_dir="./output")
# ... setup trainer ...
trainer.init()

# Load from checkpoint
trainer.load_state("./output/checkpoints/step_1000")

# Continue training
trainer.train()
```

## API Reference

### AnyEMA

| Method | Description |
|--------|-------------|
| `__init__(named_parameters, scale=1.0, capture=[])` | Initialize EMA with model parameters |
| `update(named_parameters, decay=0.999)` | Update EMA parameters |
| `swap(named_parameters)` | Swap model parameters with EMA parameters |

### AnyTrainer

| Attribute | Description |
|-----------|-------------|
| `output_dir` | Directory for outputs and checkpoints |
| `models` | List of trainable models |
| `non_trainable_models` | List of models used but not trained |
| `optimizer` | PyTorch optimizer |
| `scheduler` | Learning rate scheduler (optional) |
| `train_dataloader` | Training data loader |
| `eval_dataloader` | Evaluation data loader (optional) |
| `batch_size` | Batch size |
| `epochs` | Number of training epochs |
| `gradient_accumulation_steps` | Gradient accumulation steps |
| `mixed_precision` | Mixed precision mode ("no", "fp16", "bf16") |
| `save_checkpoint_every_steps` | Checkpoint saving frequency |
| `eval_every_steps` | Evaluation frequency |
| `max_grad_norm` | Maximum gradient norm for clipping |

| Method | Description |
|--------|-------------|
| `init()` | Initialize the trainer |
| `train()` | Start training loop |
| `train_step(step, batch, device, weight_dtype)` | Override to define training step |
| `eval_step(step, batch, device, weight_dtype)` | Override to define evaluation step |
| `eval_begin(step)` | Called before evaluation |
| `eval_end(step)` | Called after evaluation |
| `gradient_sync(step)` | Called after gradient synchronization |
| `save_state(directory)` | Save trainer state to directory |
| `load_state(directory)` | Load trainer state from directory |

## License

MIT License
