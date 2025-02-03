"""
Welcome to the "Building GPT-2 from Scratch" Python Course!

In this course, we’ll dive deep into the fascinating world of natural language processing (NLP) and machine learning by building the GPT-2 model from the ground up. GPT-2, one of the most powerful transformer-based models, is known for generating coherent and contextually relevant text, making it a popular choice for a variety of NLP applications.

Throughout this course, you’ll learn key concepts like:

- How transformers work and the mechanics behind the attention mechanism
- The architecture of GPT-2 and how it's designed for autoregressive text generation
- Step-by-step implementation of GPT-2, from tokenization and embedding layers to the final output layers
- The practicalities of training and fine-tuning a model of this scale
- Key challenges in working with large models, like memory optimization and efficient computation

By the end of this course, you’ll not only understand the intricacies of GPT-2 but also be equipped to build, train, and customize your own generative models. Whether you’re new to deep learning or looking to deepen your understanding, this course will provide you with the tools and insights needed to take on advanced NLP projects.

So, let’s get started on this exciting journey to building one of the most cutting-edge language models from scratch!
"""


import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
import inspect
import sys
import tiktoken
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimensions


class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config. n_embd % config.n_head == 0
        # key, query, value projections for all heads, in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularisation
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # mask not bias, but following OpenAI naming conventions
        self.register_buffer("bias", torch.tril(torch.ones(
            config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # batch size, sequence length, embedding dimensions
        B, T, C = x.size()
        # calculate query, key, value for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        # Using fused kernel with flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # Gaussian Error Linear Unit
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f=nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02  # default
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        It defines the forward pass of the model, which takes an input tensor idx (token indices) and returns the model's output (logits).
        idx is a tensor of shape (B, T), where:
        B is the batch size (number of sequences in the batch).
        T is the sequence length (number of tokens in each sequence).
        """
        B, T = idx.size()
        assert T <= self.config.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # Loading pre trained gpt-2 model weights from hugging face
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print("Loading weights from pre-trained gpt : %s" % model_type)

        # n_layer, n_head, n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124 M params
            # 250 M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            # 774 M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            # 1.558 B params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        # Initialize mini GPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # Initialize hugging face transformer model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy hugging face model tensors
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(
            '.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # the following weights are transposed in hugging face GPT-2 as it uses
        # Conv1d for qkv projection. However, since we are using linear functions,
        # we have to manually transpose them back.

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"Mismatched keys : {
            len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"num decayed param tensors : {
              len(decay_params)}, with {num_decay_params} parameters")
        print(f"num non-decayed param tensors : {
              len(nodecay_params)}, with {num_nodecay_params} parameters")

        fused_available = "fused" in inspect.signature(
            torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"Using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(
            0.9, 0.95), eps=1e-8, fused=use_fused)

        return optimizer


# -----------------------------------------------------------------------------------
# dataloader

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)}")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        # state
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position+B*T + 1]
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)  # targets

        self.current_position += B * T * self.num_processes

        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank

        return x, y


# -----------------------------------------------------------------------------------
"""
Training loop script

This script uses distributed data parallel technology provided by pytorch to utilize multiple
GPUs if we have them.
"""

# DDP launch script for multiple GPUs
# Also not supported on Windows or MacOS. Has to be done on linux
# torchrun --standalone --nproc_per_node={number_of_GPUs} gpt2.py

# check if this is a ddp run
ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f"cuda: {ddp_local_rank}"
    torch.cuda.set_device(device)
    # for logging
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = 'mps'

print(f"Using device : {device}")


# ----------------------------------------------------------------------------------

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


# simulate 0.5 M batch size
total_batch_size = 524288
B = 8  # micro batch size
T = 1024  # sequence length
assert total_batch_size % (B * T * total_batch_size) == 0
grad_accum_steps = total_batch_size // (B * T * total_batch_size)
if master_process:
    print(f"Total desired batch size : {total_batch_size}")
    print(f" => Calculated gradient accumulation steps: {grad_accum_steps}")

"""
Change from normal float32 to TF32 as it truncates the last 13 bits from
mantissa thus costing a little bit of precision, but significantly improving
the calculation time.
"""
# convert FP32 to TF32
torch.set_float32_matmul_precision('high')

# create model
train_loader = DataLoaderLite(
    B, T, process_rank=ddp_rank, num_processes=ddp_world_size)

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# torch compile increases compilation time, but model becomes faster. However,
# it is only supported on linux since it requires triton, which is only
# available on linux. So we can't use this on windows or mac.
# model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Assign ddp model while using ddp
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = 0.1 * max_lr
warmup_steps = 10
max_steps = 10


def get_lr(it):  # learning rate schedule
    # linear warmup to highest lr
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # min lr for final edge case
    if it > max_steps:
        return min_lr
    # use cosine decay from initial to final edge cases
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# optimization
optimizer = raw_model.configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, device=device)
init_time = time.time()
final_time = 0
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        # normalizer for loss
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        # disable GPU sync in every step
        # we only want to sync at the last step
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == grad_accum_steps - 1)
        loss.backward()

    # we want average of all loses instead of loss for every single parallel process
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # clip the global gradient norm at 1.0
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # dynamic learning rate per iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()  # wait for gpu to finish work
    t1 = time.time()
    final_time = t1
    dt = t1 - t0  # time delta in milliseconds
    processed_tokens = train_loader.B * \
        train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = (processed_tokens) / dt
    if master_process:
        print(f"step {step:4d} | loss : {loss_accum.item():10.3f} | lr : {lr:.4e} | norm : {
            norm:10.3f} | dt : {dt*1000:.2f}ms | tok/sec : {tokens_per_sec:.2f}")

total_time = (final_time - init_time)
print(f"Total computation time : {total_time:.2f} seconds")

# destroy parallel process groups before exit
if ddp:
    destroy_process_group()


sys.exit(0)


# Create prefix tokens to give our model a start of sentence
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I am a languge model, ")
# tokens = torch.tensor(tokens, dtype=torch.long)  # shape (8, )
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # shape (5, 8)
# x = tokens.to(device)

# Generate from our model
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# Keep generating tokens until the sequence reaches max_length
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)[:, -1, :]  # Get logits for the last token only
        probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
        # Sample from top 50 tokens based on their probabilities
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # Sample one token from the top 50
        ix = torch.multinomial(topk_probs, 1)
        # Select the corresponding token from topk_indices
        x = torch.cat((x, topk_indices.gather(-1, ix)), dim=-1)


# Print the generated text
tokens_list = x[:, :max_length].tolist()
decoded_list = [enc.decode(tokens) for tokens in tokens_list]
for decoded in decoded_list:
    print(">", decoded)


# 2:50:00
