from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
import matplotlib.pyplot as plt
import numpy as np

import math
import os
import time

from hellaswag import render_example, iterate_examples

device ='cuda' if torch.cuda.is_available() else 'cpu'
master_process = True
print('device:', device)

torch.set_float32_matmul_precision('high')
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

B = 64 # micro batch size: tek forward/backward'daki sequence sayısı
T = 1024 # sequence length
total_batch_size = 524288 # 2**19 ~0.5M token (GPT-3 Small'un batch size'ı)
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

print("cuBLAS TF32:", torch.backends.cuda.matmul.allow_tf32)
print("cuDNN TF32:", torch.backends.cudnn.allow_tf32)

# Tek katmanda birçok attention head'i bulunur.
# matematiksel hesaplamalar aynıdır ama GPU için işleri kolaylaştırır.
class CasualSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding size

        # nh -> "number of heads"
        # hs -> "head size"

        # .split() den çıkan 3 tensor daki elemanlar doğrudan qkv tensor'undaki elemanlara bağlıdır.
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, 2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.c_proj(y)

        return y
        

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x
        
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

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

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying (ağırlık paylaşımı)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)


    # Weight Initialization

    # Modelin ağırlıklarını ortalaması 0 ve standart sapması 0.02 olan normal dağılımla başlatıyoruz.
    # Böylece ağırlıklar başlangıçta küçük ve dengeli değerler alarak modelin
    # eğitim sürecine uygun bir başlangıç yapmasını sağlıyoruz.
    # Bias değerlerini ise 0 olarak initialize ediyoruz.
    #
    # Transformer mimarisinde residual bağlantılar nedeniyle katmanlar boyunca
    # aktivasyonların büyümesini kontrol etmek önemlidir. Bunun için residual
    # projection katmanlarına ayrıca özel initialization uygulanabilir. 
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        """
        Weight Decay

        2 boyutlu tensor lerin ağırlıklarını günceller.
        """
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
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


# from transformers import GPT2LMHeadModel
# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model.eval()
model.to(device)
# model = torch.compile(model)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073
def get_lr(it):
    """
    Cosine Learning Rate Scheduler with Linear Warmup

    en başta warmuo_steps adım kadar learning rate'i yavaşça max_lr seviyesine kadar getiriyor
    sonra max_steps adım kadar learning rate'i yavaşça düşürüyor
    """
    if it < warmup_steps:
        return (max_lr * (it+1) / warmup_steps)

    if it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (min_lr + coeff * (max_lr - min_lr))

sd_hf = model.state_dict()

for k,v in sd_hf.items():
    print(k, v.shape)

print("total parameters:", sum(p.numel() for p in model.parameters()))

idx = torch.randint(255, (1, 1024), dtype=torch.long, device=device)
idx = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long, device=device)
logits, loss = model(idx)
print(logits)

num_return_sequences = 5
max_length = 50

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a newly developed language model.")
tokens = torch.tensor(tokens, dtype=torch.long, device=device)

print("len of tokens:", len(tokens))

x = tokens.unsqueeze(0).repeat(num_return_sequences, 1)

# while x.size(1) < max_length:
#     with torch.no_grad():
#         logits, loss = model(x)
#         logits = logits[:, -1, :]
#         probs = F.softmax(logits, dim=-1)

#         # burada en yüksel 50 olasılıkla beraber olasılıkların tensordaki indeksleri döndürülür.
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

#         # burada en yüksek olasılıklardan birisi seçilir.
#         ix = torch.multinomial(topk_probs, 1)

#         # burada seçilen olasılığın orijinal indeksi yer alır. (vocab_size baz alınarak)
#         xcol = torch.gather(topk_indices, -1, ix)

#         # sonucu ekleme işlemi
#         x = torch.cat((x, xcol), dim=1)

print(x)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T

        assert split in {'train', 'val'}

        # dir_path = os.path.dirname(os.path.realpath(__file__))
        # with open(dir_path + '/input.txt', 'r') as f:
        #     text = f.read()

        # enc = tiktoken.get_encoding('gpt2')
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens, device=device)

        # self.current_position = 0
        
        # print(f"loaded {len(self.tokens)} tokens")
        # print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        data_root = "dataset"

        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")

        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T

    # def next_batch(self):
    #     B, T = self.B, self.T
    #     buf = self.tokens[self.current_position : self.current_position + (B*T) + 1]
    #     x = buf[:-1].view(B, T) # inputs
    #     y = buf[1:].view(B, T) # targets

    #     self.current_position += B*T

    #     if self.current_position + (B * T + 1) > len(self.tokens):
    #         self.current_position = 0

    #     return (x, y)

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T
        return x, y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------

train_loader = DataLoaderLite(B=B, T=T, split='train')
val_loader = DataLoaderLite(B=B, T=T, split='val')

# x, y = train_loader.next_batch()
# for i in range(x.size(0)):
#     print(f"Sequence {i}:")
#     print(enc.decode(x[i].tolist()))
#     print("-" * 50)

# betas = first momentum, second momentum
# gradient boyutları birbirlerinden büyük değişkenlik gösterebilir.
# daha stabil şekilde gradient değerleri almak için momentum'ları kullanıyoruz
# örn: %90 oranında geçmiş gradientların ortalamasını al + %10 oranında yeni gradient in değerini al
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)


o = open('o.txt', 'w+')

# create the log directory we will write checkpoints to and log to
# play.ipynb bu dosyayı okuyup train/val loss ve HellaSwag grafiklerini çiziyor
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

times = []
lrs = []
outs = []
for i in range(max_steps):
    t0 = time.time()


    # validation loss 
    if i % 100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{i} val {val_loss_accum.item():.4f}\n")


    # once in a while evaluate hellaswag
    if i % 100 == 0:
        model.eval()
        num_correct_norm = 0
        num_total = 0
        # Karpathy burada "for i, example in enumerate(...)" kullanıyor, bizde i adım sayacı olduğu için
        # enumerate'i kaldırdık. Tek GPU'da çalıştığımız için ddp_rank filtresine de gerek yok.
        for example in iterate_examples("val"):
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{i} hella {acc_norm:.4f}\n")


    if i > 0 and i % 100 == 0:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Merhaba, ben bir dil modeliyim.")
        xgen = torch.tensor(tokens, dtype=torch.long, device=device)
        xgen = xgen.unsqueeze(0).repeat(num_return_sequences, 1)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                logits, loss = model(xgen)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)

                # burada en yüksel 50 olasılıkla beraber olasılıkların tensordaki indeksleri döndürülür.
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

                # burada en yüksek olasılıklardan birisi seçilir.
                ix = torch.multinomial(topk_probs, 1)

                # burada seçilen olasılığın orijinal indeksi yer alır. (vocab_size baz alınarak)
                xcol = torch.gather(topk_indices, -1, ix)

                # sonucu ekleme işlemi
                xgen = torch.cat((xgen, xcol), dim=1)

        for _ in range(num_return_sequences):
            tokens = xgen[_, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"sample {_}: {decoded}")


    # training looğ
    optimizer.zero_grad()
    model.train()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # import code; code.interact(local=locals())

        # Gradient Accumulation
        # backward() her çağrıldığında gradient'ler .grad'ın üzerine eklenir (+=).
        # F.cross_entropy tek bir micro batch'in ORTALAMA loss'unu verir; bölmeden backward yaparsak
        # grad_accum_steps tane ortalamanın TOPLAMINI almış oluruz (gradient grad_accum_steps kat büyür).
        # loss'u grad_accum_steps'e bölünce toplanan gradient, 524288 token'lık büyük batch'in
        # tek seferde hesaplanan ortalama loss'unun gradient'ine eşit olur.
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    lrs.append(lr)
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) # saniye olarak zaman farkı (log'da ms'ye çevriliyor)
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    times.append(dt)
    log_str = f"step {i} | loss: {loss_accum.item():.6f} | lr: {optimizer.param_groups[0]['lr']:.4e} | norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
    print(log_str)
    if master_process:
        with open(log_file, "a") as f:
            f.write(f"{i} train {loss_accum.item():.6f}\n")
    if i < 50:
        outs.append(log_str)

plt.plot(lrs)
plt.xlabel('step')
plt.ylabel('learning rate')
plt.title('Cosine LR schedule with linear warmup')
plt.savefig('lrs_graph.png', bbox_inches='tight')

o.write('\n'.join(outs))

import sys
sys.exit()