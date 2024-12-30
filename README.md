# Reproducing GPT-2 (124M)

Notes taken, with references and additional observations, while following along Karpathy's GPT-2 (124M)

## LayerNorm before/after blocks

Why in GPT-2 the LayerNorm is before the blocks unlike the Attention paper where they are after the blocks?

TLDR: **Pre-normalization** - Clean residual pathway is desirable from an optimization perspective.

- If LayerNorm after the block (FF layer), then the residual stream will have the normalized values.
- Instead, we want a clean residual stream. 
- Because addition distributes gradients, thus in a clean residual stream the gradients from the top flow straight to the inputs, through the residual pathway unchanged, (and also through the blocks where they have their contributions).
- This is nice and efficient from a training perspective (ResNet paper).


## Transformer Blocks as Map-Reduce ops
  
`Attention` is a:
- communication
- weighted sum
- aggregation
- pooling
- `reduce`... function

`MLP` is a:
- every single token individually
- no message passing between tokens
- `map` ... function


## GeLU
- paper - stochastic regularizers ... expectation of a modification to Adaptive Dropout...
- tanh - historical version due to earlier slower in tf

Intuitively:
- dead ReLU neuron problem (earlier video)
- in the tail of the ReLU, if any of the activations fall there, there is no change or adaptation.
- Whereas in GeLU there is always a grad contrib and there is always change.

## Logit calculation - sanity check
When using NLL loss, at init the theoritical loss should be $$-\log(\frac{1}{\text{num classes}})$$ since uniform random distribution (prob of each class is $\frac{1}{\text{num classes}}$).

here, $$-\log(\frac{1}{\text{vocab size}})$$
$$=-\log(\frac{1}{\text{50257}})$$
$$=10.825$$

So, the first NLL loss (cross entropy without softmax) calc just after random init should be ~10.825.

## GPT-2 Weight Sharing

LM Head == WTE (token embedding) [even the data_ptrs are the same]

Refs:
- [GPT-2 original code](https://github.com/openai/gpt-2/blob/master/src/model.py) - logits = tf.matmul(h_flat, `wte`, transpose_b=True)
- [Attention is all you need](https://arxiv.org/pdf/1706.03762) -> Ref[30]
    > In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [ 30 ].

- Ref[30] -> [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
    > We call U the input embedding, and V the output embedding. In both matrices, we expect rows that correspond to similar words to be similar: for the input embedding, we would like the network to react similarly to synonyms, while in the output embedding, we would like the scores of words that are interchangeable to be similar (Mnih and Teh, 2012).

Intuitively, for the input end, similar tokens should be closer in embedding space, and for the output end, similar tokens should have similar probabilities. Essentially, in both cases, similar tokens should have similar embeddings or similar weights. [30] also finds that output embeddings behave like word-embeddings.

Also, this a LOT of params! 

768 * 50257 = ~38M params, which is 30% of the model size (124M).

And, due to weight sharing we don't have to train these extra params so faster training.

## Weight Initialization
- GPT-2 weights:
    - normal distribution with mean = 0.0 and std dev = 0.02
    - positional enc -> normal dist with mean = 0.0 and std dev = 0.01
    - here, we keep all std dev = 0.02
- Bias - init to zero
- LayerNorm - PyTorch default -> scale = 1.0, offset = 0

### Why Std Dev = 0.02
According to the Xavier init scheme, std dev $= \frac{1}{\sqrt(\text{num incoming features})}$.

Here, 

$\frac{1}{\sqrt(\text{768})} \approx 0.03$

$\frac{1}{\sqrt(\text{1600})} \approx 0.025$

$\frac{1}{\sqrt(\text{3 x 1600})} \approx 0.014$

## Std Dev grows inside the residual stream

See nb code block.

- Scale weights of residual layers at init by $\frac{1}{\sqrt(N)}$, N = num residual layers.

Assuming indeppendent and all variances are equal,

$Var(\sum^{N}_{i=1} i) = \sum^{N}_{i=1} Var(i) = N Var(i_0)$ 

and, $\sum^{N}_{i=1} Var(i) = N \sigma(i_0)^2$ or $\sigma(\sum^{N}_{i=1} i) = (\sqrt(N)) \sigma(i_0)$

Scaled version, dividing each dist by $\sqrt(N)$, scales $\sigma(i)$ by $\sqrt(N)$ ie.e variance by $N$.

$= \sqrt(\sum^{N}_{i=1} Var(\frac{1}{N} i))= \sqrt(\frac{N}{N} Var(i_0)) = \sigma(i_0)$

So, this is a way to `control the growth of the activations` in the residual stream.

## Optimizations for higher training throughput

### Batch Size and Tokens/sec
---
- Increase batch size, duh!
- However, time taken at each step might increase, so look at tokens processed per second.


### FP Precision
---
Throughput vs Precision

Only works in Ampere GPUs, else performance is affected (at least in Turing GPUs).

- By default, float32 
- FP16 and BF16 require Tensor Cores (4x4 matrix multiply) (See A100 Tensor Core Arch.)

- TF32 - same range as FP32 - `torch.set_float32_matmul_precision("high")`
    - still memory b/w bound due to data moving around.

- If FP16, then use grad scalers (coz range smaller than FP32) [grad scaler](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-gradscaler)
- use BF16 (if available) (See AMP recipe)
    - Do not call bfloat16() when using autocast and only surround till loss calculation, leave the backward and optimization step as is. [autocast](https://pytorch.org/docs/stable/amp.html#autocasting)

Mixed Precision as ([ops affected](https://pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float16)):

    - loss.dtype and params (weights) -> torch.float32
    - logits.dtype (activations) -> torch.bfloat16

Refs:
- [AMP Recipe](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
- [NVIDIA A100 Tensor Core GPU Architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)

- [NVIDIA AMPERE GA102 GPU ARCHITECTURE](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)

### torch.compile()
---
- uses [Triton](https://github.com/triton-lang/triton) - official builds only for Linux.
- doesn't run in eager mode (or layer by layer)
- [reduces Python overhead and GPU read/writes](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#:~:text=Speedup%20mainly%20comes%20from%20reducing%20Python%20overhead%20and%20GPU%20read/writes)
- things break in Python 3.9 in RedHat ¯\\_(ツ)_/¯

main optimizations torch.compile will perform are:
- kernel fusions, thus reducing read/writes from GPU cache to HBM
- CUDA graphs
- Template matching for SDPA (i.e: flash attention) - template is key!

=> (with all other optims) 8.8s -> 7.5s = 15% improvement!

### Flash Attention
---
[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135) (2022)

    FlashAttention does not read and write the large N x N attention matrix to HBM, resulting in an 7.6x speedup on the attention computation.

- uses [online softmax trick](https://arxiv.org/pdf/1805.02867) (2018)
- calculates softmax in a streaming/online manner

Essentially,
- Flops doesn't matter; memory access pattern matters, which impacts the throughput.
- There might be more optimizations as such which torch.compile() (or triton) can't find.

```python
y= F.scaled_dot_product_attention(q, k, v, is_causal=True)
```
Note: Without Ampere GPU (like Turing GPU), slight improvement (2.5%) which may be due to better algorithmic re-write (such as using bitwise ops) in SDPA, rather than FlashAttention. However, exact same values.

- Turing: 2.5% (dt: 434.25ms -> 423.98ms | tok/sec: 2359.14 -> 2413.07)
- Ampere: ~27%

Refs
- [PyTorch SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

### Powers of 2 - nice numbers
--- 
- Increasing the vocab_size from 50,257 ( 1100010001010001 ) to 50304 ( 1100010010000000 ).
- Does not break anything since these extra tokens are never used - but the network has to learn that probs for these tokens have to be driven to zero, same as other unused tokens.
- FLOPS increases but throughput increases
- Because kernels for boundary conditions can be inefficient and lead to an overhead, which is minimized if the matrices nicely fits in the PEs.
- Roughly 4% improvement (for nightly) and ~30% (<2.3.1) (as in video) - for Turing GPUs, without above FP opts and compile, stays the same.

### Hyperparameters
---
From GPT-3 (where only 2048 context window is the only change)

- AdamW (Adam with weight decay bugfix) 
    - `betas = (0.9, 0.95)` (default (0.9, 0.999))
    - `eps = 1e-8` (same as default)
    - `Weight decay = 0.1` (default Adam is 0.01)
        - Selectively as below
        - Params to be weight-decayed: weights for matmul (>2-dim tensors)
        - Not to be weight-decayed: biases, LayerNorm (scales, biases) (1-dim tensors)
    - `fused = True` - if available, faster than for-loop.

- Gradient Clipping
    - clip the global norm of the gradient at 1.0
    ```python
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    ```
    - Why? Sometimes during training we might get unlucky and get a "bad batch" where the gradients are extremely large. This can "shock" the model. Essentially, a way to not fluctuate training due to anomalous batches.
    - If the norm is stable over steps, things are good.
    - If the norm is climbing, training is destabilizing.

- Learning Rate Scheduler
    - Cosine decay learing schedule
    - linear warmup for 1.5% of tokens
    - cosine decay down to 10% of original value
    - can use torch LR Scheduler, but write five lines of code!
    ```python
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    ```

- Batch size increase
    - not very useful as not an algorithmic improvement, but more of a systems improvement.
    - initially use a smaller batch size and linearly increase to full batch size.
    - works because initially the model is learning to ignore most of the tokens and learn the low-level biases of the input text. 
    - these tokens appear, these don't -> grads are very similar!
    - so a large batch does contribute much and thus can be reduced to speed up training.

- Sample data without replacement until next epoch
    - since batches are iterated over the text in sequence, this is already implemented.

### Grad Accum

Reproducing paper (batch sizes w/ same hyperparams) without enough GPU memory!

Why same "total batch size"? - because the hyperparams are **correlated** with each other.

```python
loss_accum = 0.0
net.zero_grad()
for micro_step in range(grad_accum_steps):
    yhat = net(x[i])
    loss = torch.nn.functional.mse_loss(yhat, y[i])
    loss = loss / grad_accum_steps # = C --- NOTE : without this 1/C is lost!
    # loss is the loss for each micro batch
    # do loss_accum for total loss for grad_accum_steps -> loss_accum after norm
    # since loss / C => grad = d(loss/C)/dx = (1/C) * dloss/dx
    loss_accum += loss.detach()
    loss.backward()
```

