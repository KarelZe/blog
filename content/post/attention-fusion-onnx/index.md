---
title: The good, the bad, the brittle. Fusing Attention in ONNX graphs✨
date: 2025-06-07T08:32:00+02:00
Description: Fusing Attention in ONNX graphs✨
Tags: [attention, self-attention, onnx, onnxscript, bart, swin]
Categories: [ml]
DisableComments: false
---

The attention mechanism is the core of Transformer-based models. Being compute-intensive, we often want to optimize it to achieve high throughput and low inference times. This article explores different approaches to optimize attention mechanism for huggingface transformers in onnx graphs.

## Background

Odds are that when working with Transformers, you come back to huggingface's Transformer package. Transformers uses *custom modelling code* for the attention layers (see e.g., [here](https://github.com/huggingface/transformers/blob/ebeec13609b537f9c760292354118c9d1d63f5a0/src/transformers/models/bart/modeling_bart.py#L147)). During export the pytorch modelling code gets first translated into an [onnx-ir representation](https://github.com/onnx/ir-py), optimized (optional), and then serialized as protobuf.[^1]

Similar to the transformers modelling code, the onnx graph will consist of low-level onnx primitives like `MatMul`, `Reshape` or `Concat` to model attention, despite the availability of highly-optimized [`Attention`](https://onnx.ai/onnx/operators/onnx__Attention.html) ops in recent versions of onnx.

```yaml
# insert sample image from netron here.
```

## Core idea of attention fusion

The core idea is now to identify patterns in the onnx graph, that look like the attention mechanism and replace the entire subgraph with the `Attention` op plus required adaptions. Fusing Attention is desirable for three reasons:

1. Its likely faster. When the onnx graph is executed on a gpu, the model is first loaded into high-bandwidth memory (HBM). Each operator is run as a small kernel (or program) on the gpu and launched into small but fast static RAM (SRAM) and then results are saved back to HBM. Memory transfers are typically slow, therefore we want to fuse multiple kernels or ops into larger kernels to minimize the amount of memory transfers required. [^2]
1. Hardware vendors often implement optimized implementations for onnx execution providers e.g., attention, which will give you an additional performance edge.
1. Operator fusion will keep your graphs tidy and thereby your inner [Marie Kondo](https://knowyourmeme.com/photos/2247427-does-it-spark-joy) happy.🤓

## Export onnx graph

Let's look at a simple bart encoder first. Despite being conceptually simple, it makes up for an interesting example, as being the cornerstone to more complex models like openai's whisper. We first prepare the model for export. I export via `torch` for ache of flexibility, while you could also use [optimum](https://huggingface.co/docs/optimum/index) if you favour higher level abstractions.

```python
import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.bart.modeling_bart import BartSdpaAttention

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "hf-internal-testing/tiny-random-bart"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.eval()


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        outs = self.encoder(input_ids, attention_mask)
        return outs["last_hidden_state"]


model = EncoderWrapper(encoder=model.model.encoder)
print(model)

text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

input_names = ["input_ids"]
output_names = ["encoder_output"]

onnx_path = "bart_model.onnx"

print(model)

torch.onnx.export(
    model,
    (input_ids,),
    onnx_path,
    export_params=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "encoder_output": {0: "batch_size", 1: "sequence_length"},
    },
    opset_version=20,
    # NOTE: deprecated in latest versions (torch > 2.6) https://docs.pytorch.org/docs/stable/onnx_torchscript.html
    export_modules_as_functions={BartSdpaAttention},
)
```

Some aspects deserve more explanation. We wrap the encoder as a class (`EncoderWrapper`) to only retrieve and rearrange inputs and outputs required for later processing. I export the scaled-dot product attention as an onnxscript function for easier fusion. This feature, however, has been deprecated in recent nightly builds of pytorch and function-based rewrites have been removed from `onnxscript`.

## Fusion with onnxruntime

```python
optimization_options = FusionOptions("bart")
optimization_options.enable_attention = True

m = optimizer.optimize_model(
    onnx_path,
    model_type="bart",
    num_heads=0,
    hidden_size=0,
    opt_level=2,
    use_gpu=False,
    verbose=True,
    optimization_options=optimization_options,
    only_onnxruntime=False,
)

optimized_path = "bart_encoder_optimized.onnx"
m.save_model_to_file(optimized_path)

print(f"Optimized ONNX model saved to {optimized_path}")
print(m.get_fused_operator_statistics())

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
encoder_outs_original = sess.run(["encoder_output"], {"input_ids": input_ids.numpy()})

sess_optimized = ort.InferenceSession(
    optimized_path, providers=["CPUExecutionProvider"]
)
encoder_outs_optimized = sess_optimized.run(
    ["encoder_output"], {"input_ids": input_ids.numpy()}
)

abs_diff = np.amax(encoder_outs_original[0] - encoder_outs_optimized[0])
print("abs_difference", abs_diff)
```

## Fusion with rule in onnxscript

```python
# example adapted from:
# https://github.com/justinchuby/onnx-meetup-2025/blob/main/demos.ipynb

import onnxscript.rewriter as rewriter


class CombineQKVWeights(rewriter.pattern.RewriteRuleClassBase):
    def pattern(cls, op, input, q_weight, k_weight, v_weight):
        q = op.MatMul(input, q_weight)
        k = op.MatMul(input, k_weight)
        v = op.MatMul(input, v_weight)
        return op.Attention(q, k, v, q_num_heads=1, kv_num_heads=1)

    def rewrite(cls, op, input, q_weight, k_weight, v_weight):
        qkv_weight = op.initializer(
            ir.tensor(
                np.concatenate(
                    [
                        q_weight.const_value.numpy(),
                        k_weight.const_value.numpy(),
                        v_weight.const_value.numpy(),
                    ],
                    axis=1,
                )
            ),
            name="qkv_weight",
        )
        combined_matmul = op.MatMul(input, qkv_weight)
        new_q, new_k, new_v = op.Split(
            combined_matmul, axis=2, num_outputs=3, _outputs=3
        )
        return op.Attention(new_q, new_k, new_v, q_num_heads=1, kv_num_heads=1)


model = build_model()
# Create the rewrite rule
rule = CombineQKVWeights.rule()
# Apply the rewrite rule to the model
rule.apply_to_model(model)
# Clean up and run shape inference. Note that you can use the Sequential pass to chain multiple passes together.
ir.passes.Sequential(
    common_passes.RemoveUnusedNodesPass(),
    common_passes.ShapeInferencePass(),
)(model)

print(model)
```

## Fusion with custom rule in onnxscript

## Performance results

## Conclusion

While researching for this blog post, I made several open-source contributions.

## References:

[^1]: see [https://github.com/justinchuby/diagrams/](https://github.com/justinchuby/diagrams/pull/1/files) for a helpful diagram on the internals of the pytorch onnx exporter.

[^2]: see [talk on large language model inference with ONNX Runtime by Kunal Vaishnavi](https://youtu.be/jrIJT01E8Xw?feature=shared&t=420)
