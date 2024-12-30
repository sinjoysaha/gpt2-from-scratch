import math, time, sys, code, inspect, tqdm
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# ---------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # since n_embd (C) = n_head (ns) * h_size (hs) as below
        assert config.n_embd % config.n_head == 0
        #
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection - needed? n_embd -> n_embd?
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # flag for scaling std dev in residual stream
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization ??
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not a bias, but mask, but following OpenAI naming
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embed dim (n_embd)

        # merged karpathy code - check Head and MultiHead code
        # calc q, k, v forr all heads in a batch and move head forward
        # nh - #heads, hs - head size, C - #channels = nh * hs
        # GPT-2 (124M) -> n_head = 12, hs = 64, so nh*hs=768 channels in transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, ns, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # attention -> "materializes" large (T, T) matrix for all q and k
        # and normalization w/ d_keys to keep variance 1

        # SDPA (FlashAttention) replaces these lines
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # # after softmax -inf -> 0
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # SDPA (FlashAttention)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # re-assemble all head o/p side by side - concat operation
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        # flag for scaling std dev in residual stream
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.c_gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTCofig:
    block_size: int = 1024
    vocab_size: int = 50257
    # #tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftoken|>
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768  # 12 * 64


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, "NANOGPT_SCALE_INIT"):
            # attn and mlp, so 2 times
            std *= (2 * self.config.n_layer) ** -0.5

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # LayerNorms -> default init - scale = 1.0, offset = 0.0

    def forward(self, idx, targets=None):
        # idx shape = (B, T)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward seq len of {T}, with model block size {T}"
        # forward token and pos embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward through each block of transformer
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface.
        For now, only focusing on gpt2 (124M)

        Args:
            model_type (_type_): _description_
        """
        assert model_type in {"gpt2"}
        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt: {model_type}")

        # n_layer, n_head and n_embd from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        }[model_type]
        config_args["vocab_size"] = 50257  # for all GPT-2 models
        config_args["block_size"] = 1024  # for all GPT-2 models
        # create from-scratch GPT-2 model
        config = GPTCofig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # ignore mask / buffers (not params)
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        # init HF/transformers GPT-2 model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        # ignore masks/buffers
        sd_keys_hf = [
            k
            for k in sd_keys_hf
            if not (k.endswith(".attn.masked_bias") or k.endswith(".attn.bias"))
        ]
        # openai uses Conv1D, we use vanilla Linear layer, so transpose when import
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys {len(sd_keys_hf)} != {len(sd_keys)}"
        # copy params
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # transpose Conv1D weights
                # print(
                #     f"  Con1D copy | {k:35} | sd_hf {str(sd_hf[k].shape):25} | sd {sd[k].shape}"
                # )
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())  # .t() - transpose
            else:
                # vanilla copy
                # print(
                #     f"Vanilla copy | {k:35} | sd_hf {str(sd_hf[k].shape):25} | sd {sd[k].shape}"
                # )
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, weight_decay, lr, device):
        # candiates that require grad
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # optim grps for - weights w/ and w/out decay - see doc for details
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum([p.numel() for p in decay_params])
        num_nodecay_params = sum([p.numel() for p in nodecay_params])
        print(
            f"num decayed param tensors: {len(decay_params)}, with {num_decay_params:,} params"
        )
        print(
            f"num non-decayed param tensors: {len(nodecay_params)}, with {num_nodecay_params:,} params"
        )

        # fused if cuda and fused available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
        return optimizer


# ----------------------------------------

import tiktoken


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        # at init load tokens and store in mem
        with open("tinyshakespeare.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)  # in cpu mem
        print(f"loaded {len(self.tokens)} tokens")
        print(f"tokens in {self.tokens.dtype}")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        # move curr pos by B * T
        self.current_position += B * T
        # if curr pos is OOB, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


# ----------------------------------------

# print(GPT(GPTCofig()))

# ----------------------------------------
# model = GPT.from_pretrained("gpt2")
# print("didn't crash yay!")

# ----------------------------------------
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# ----------------------------------------
enc = tiktoken.get_encoding("gpt2")
# with open("tinyshakespeare.txt", "r") as f:
#     text = f.read()
# text = text[:1000]
# tokens = enc.encode(text)
# B, T = 4, 32
# buf = torch.tensor(tokens[: B * T + 1], device=device)
# x = buf[:-1].view(B, T)
# y = buf[1:].view(B, T)

# ----------------------------------------
# Hyperparams (mostly due to grad accum code)

total_batch_size = 524288 # 2**19 ~ 0.5M tokens from GPT-3 paper

# GPU poor! This file has settings for Turing GPU.
# Following commits should have training code for 2 Ampere GPUs 
B = 8 # micro batches
T = 128 # sequence length

assert total_batch_size % (B * T) == 0, "make sure total_batch_size is div by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# ----------------------------------------
# init data loader 
train_loader = DataLoaderLite(B=B, T=T)

# ----------------------------------------
# Only works in >Ampere
# torch.set_float32_matmul_precision("high")

# ----------------------------------------
# random model init
# model = GPT(GPTCofig()) # 50257 vocab_size
model = GPT(GPTCofig(vocab_size=50304))

# ----------------------------------------

model.to(device)

# Can't torch.compile() without triton (on windows)
# start_time = time.time()
# model = torch.compile(model)
# print(f"torch.compile() time taken: {time.time() - start_time}s.")

# ----------------------------------------
# logits, loss = model(x, y)
# print(logits.shape)
# print(loss)

# ----------------------------------------
# LR scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50


def get_lr(it):
    # 1) linear warmup for warmup_steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) flat tail after max_steps
    if it > max_steps:
        return min_lr
    # 3) in between, cosine decay till min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


# ----------------------------------------

# Hyperparams from GPT-3 paper
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

# ----------------------------------------
# Using weight decay selectively and fused AdamW
optimizer = model.configure_optimizers(weight_decay=0.1, lr=6e-4, device=device)

# ----------------------------------------

# optimize!
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in tqdm.tqdm(range(grad_accum_steps)):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
    
        # only works in Ampere GPUs, else performance is affected (at least in Turing GPUs)
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)

        # normalizing for grad accum
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()

        # code.interact(local=locals())
        loss.backward()
    
    # grad norm clipping
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # set LR from schedule
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000  # time diff in ms
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / (t1 - t0)
    print(
        f"step {step} | loss: {loss_accum.item():.4f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
    )

sys.exit(0)
# ----------------------------------------
num_return_sequences = 5
max_length = 50

# prefix tokens (prompt)

# tokens = enc.encode("Hello, I'm  language model,")
# tokens = torch.tensor(tokens, dtype=torch.long)  # (8, )
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
# x = tokens.to(device)

# ----------------------------------------
model.eval()

# generate - x = (B, T)
start_time = time.time()
while x.size(1) < max_length:
    # forward
    with torch.no_grad():
        logits, loss = model(x)
        logits = logits[:, -1, :]  # last pos logits (B, vocab_size)
        probs = F.softmax(logits, dim=-1)  # logits -> probabilities
        # top-k sampling 50 (hf pipeline default)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # both (5, 50)
        # sample with the topk_probs
        ix = torch.multinomial(topk_probs, 1)  # (B, 1)
        # gather - check docs >
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # append to sequence
        x = torch.cat((x, xcol), dim=1)

# print
print(num_return_sequences, x.size(0))
for i in range(x.size(0)):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

print(f"Time taken: {time.time() - start_time}s.")

# ----------------------------------------
