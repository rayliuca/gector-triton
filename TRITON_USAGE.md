# GECToR with Triton Inference Server

This document describes how to use GECToR with NVIDIA Triton Inference Server for remote model inference.

## Overview

`GECToRTriton` is a class that extends the base `GECToR` class to support models served on a remote Triton Inference Server. Instead of loading the BERT/RoBERTa model locally, it sends inference requests to a Triton server.

## Prerequisites

1. **Triton Client Library**: Install the Triton client library:
   ```bash
   pip install tritonclient[grpc]
   ```

2. **Triton Server**: You need a running Triton Inference Server with your GECToR model deployed. See [Deploying Models to Triton](#deploying-models-to-triton) below.

## Usage

### Basic Usage

```python
from transformers import AutoTokenizer
from gector import GECToRTriton, predict, load_verb_dict
import torch

# Load configuration from a pretrained model
model = GECToRTriton.from_pretrained(
    'gotutiyan/gector-roberta-base-5k',  # Config source
    triton_url='localhost:8001',          # Triton server URL
    model_name='gector_bert',             # Model name on Triton
    model_version='1',                    # Model version
    verbose=True                          # Enable logging
)

# Load tokenizer as usual
tokenizer = AutoTokenizer.from_pretrained('gotutiyan/gector-roberta-base-5k')

# Load verb dictionary
encode, decode = load_verb_dict('data/verb-form-vocab.txt')

# Prepare input
srcs = [
    'This is a correct sentence.',
    'This are a wrong sentences'
]

# Perform prediction
corrected = predict(
    model, tokenizer, srcs,
    encode, decode,
    keep_confidence=0.0,
    min_error_prob=0.0,
    n_iteration=5,
    batch_size=2,
)

print(corrected)
```

### Alternative: Direct Initialization

```python
from gector import GECToRTriton, GECToRConfig

# Load config from a pretrained model
config = GECToRConfig.from_pretrained('gotutiyan/gector-roberta-base-5k')

# Create Triton model
model = GECToRTriton(
    config=config,
    triton_url='localhost:8001',
    model_name='gector_bert',
    model_version='1',
    verbose=True
)

# Use as normal...
```

## Deploying Models to Triton

To use `GECToRTriton`, you need to deploy your GECToR model to a Triton Inference Server. Here's a general guide:

### 1. Model Repository Structure

Create a model repository with the following structure:

```
model_repository/
└── gector_bert/
    ├── config.pbtxt
    └── 1/
        └── model.pt
```

### 2. Export Your Model

You'll need to export your GECToR model in a format compatible with Triton. For PyTorch models, you can use TorchScript:

```python
from gector import GECToR
import torch

# Load your model
model = GECToR.from_pretrained('gotutiyan/gector-roberta-base-5k')
model.eval()

# Create example inputs
example_input_ids = torch.randint(0, 1000, (1, 80))
example_attention_mask = torch.ones((1, 80), dtype=torch.long)

# Trace the model
traced_model = torch.jit.trace(
    model,
    (example_input_ids, example_attention_mask)
)

# Save the traced model
traced_model.save('model_repository/gector_bert/1/model.pt')
```

### 3. Create Model Configuration

Create `config.pbtxt` in the model directory:

```protobuf
name: "gector_bert"
platform: "pytorch_libtorch"
max_batch_size: 32

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [-1]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [-1]
  }
]

output [
  {
    name: "logits_labels"
    data_type: TYPE_FP32
    dims: [-1, -1]
  },
  {
    name: "logits_d"
    data_type: TYPE_FP32
    dims: [-1, -1]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

### 4. Start Triton Server

```bash
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v /path/to/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

### 5. Verify Server is Running

```bash
curl -v localhost:8000/v2/health/ready
```

## API Reference

### GECToRTriton

```python
class GECToRTriton(GECToR):
    def __init__(
        self,
        config: GECToRConfig,
        triton_url: str = "localhost:8001",
        model_name: str = "gector",
        model_version: str = "1",
        verbose: bool = False
    )
```

**Parameters:**
- `config`: GECToRConfig object with model configuration
- `triton_url`: URL of the Triton Inference Server (gRPC endpoint)
- `model_name`: Name of the model on the Triton server
- `model_version`: Version of the model to use
- `verbose`: Enable verbose logging

**Methods:**

All methods from the base `GECToR` class are available, but inference happens on the Triton server:

- `forward()`: Sends inference request to Triton server
- `predict()`: Performs prediction using Triton server
- `from_pretrained()`: Class method to create instance from pretrained config

**Note:** Training-related methods (`tune_bert()`, `init_weight()`) are no-ops for Triton models.

## Limitations

- **Training not supported**: The Triton model can only be used for inference. Training with labels is not supported.
- **Server dependency**: Requires a running Triton server with the model deployed.
- **Network latency**: Performance depends on network connection to the Triton server.

## Troubleshooting

### "tritonclient is required for GECToRTriton"

Install the Triton client library:
```bash
pip install tritonclient[grpc]
```

### "Triton server at {url} is not live"

Check that your Triton server is running and accessible at the specified URL:
```bash
curl localhost:8000/v2/health/live
```

### "Model {name} (version {version}) is not ready"

Verify the model is loaded on the server:
```bash
curl localhost:8000/v2/models/{model_name}/versions/{version}/ready
```

Check Triton server logs for model loading errors.

## Performance Considerations

- **Batch Size**: Adjust batch size based on your Triton server's GPU memory
- **Network**: Use a high-bandwidth, low-latency connection to the Triton server
- **Model Optimization**: Consider using TensorRT or ONNX for optimized inference on Triton
- **Concurrency**: Triton supports concurrent requests - configure instance groups appropriately

## Example: Complete Workflow

```python
from transformers import AutoTokenizer
from gector import GECToRTriton, predict, load_verb_dict

# 1. Initialize model
model = GECToRTriton.from_pretrained(
    'gotutiyan/gector-roberta-base-5k',
    triton_url='localhost:8001',
    model_name='gector_roberta',
    verbose=True
)

# 2. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('gotutiyan/gector-roberta-base-5k')

# 3. Load verb dictionary
encode, decode = load_verb_dict('data/verb-form-vocab.txt')

# 4. Read input
with open('input.txt', 'r') as f:
    srcs = f.read().strip().split('\n')

# 5. Perform corrections
corrected = predict(
    model, tokenizer, srcs,
    encode, decode,
    keep_confidence=0.3,
    min_error_prob=0.6,
    n_iteration=5,
    batch_size=32,
)

# 6. Save output
with open('output.txt', 'w') as f:
    f.write('\n'.join(corrected))
```
