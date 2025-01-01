---
title: Make DONUT go brrr🚀
date: '2025-01-01'
Description: In this post we explore how convert DONUT to onnx end-to-end.
Tags: [DONUT, transformers, onnx, inference-optimization]
Categories: [machine-learning]
DisableComments: false
---

I had several weeks off from work and wanted to do a macro cycle to ramp up my onnx skills[^1]. *Oh boy* did I go on a journey!

## DONUT architecture

TODO:

## converting image processor

TODO:

## converting the tokenizer

TODO:

## converting the encoder

TODO:

## converting the decoder with kv cache

TODO:

## merging the image processor, encoder, decoder without past

TODO:

## implementing beam search

TODO:

## putting it all together

TODO:

## final session

```python
so = ort.SessionOptions()
so.register_custom_ops_library(get_library_path())
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

max_length = 4096
min_length = 1
num_beams = 1
num_return_sequences = 1
repetition_penalty = 0.0

sess_donut = ort.InferenceSession(
    output_dir / "donut_e2e.onnx",
    sess_options=so,
    providers=["CPUExecutionProvider"],
)
sess = {
    "pixel_values": np.array(sample_img),
    "max_length": np.array([max_length], dtype=np.int32),
    "min_length": np.array([min_length], dtype=np.int32),
    "num_beams": np.array([num_beams], dtype=np.int32),
    "num_return_sequences": np.array([num_return_sequences], dtype=np.int32),
    "length_penalty": np.array([0.01], dtype=np.float32),
    "fairseq": [True],
}
sess.run(None, ort_inputs)[0]
```

## go really brr...

TODO: add more optimization tricks

[^1]: Inspired by this [video](https://www.youtube.com/watch?v=SgaN-4po_cA).
