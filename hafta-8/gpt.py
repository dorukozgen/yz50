import torch
import torch.nn as nn
from torch.nn import functional as F

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

from model import BigramLanguageModel

batch_size = 32
max_iters = 10000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
block_size = 8

torch.manual_seed(1337)

def get_batch(split):

    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return (x, y)

# xb, yb = get_batch('train')
# print(xb.shape)
# print(yb.shape)
# print(xb[1])
# print(yb[1])

# print('-----')

# for b in range(batch_size):
#     for t in range(block_size):
#         context = xb[b, :t+1]
#         target = yb[b, t]
#         # print(f"input: {context.tolist()} target: {target}")

if __name__ == "__main__":
    with open(dir_path + './data/input.txt', 'r', encoding='utf-8') as file:
        text = file.read()
        print(len(text))

    # Metinde yer alan tüm karakterlerin listesi ve uzunluğu
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(''.join(chars))
    print(vocab_size)

    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    print(encode("hello, how are you?"))
    print(decode(encode("hello, how are you?")))

    # embedding dizisini vektöre dönüştürme
    data = torch.tensor(encode(text), dtype=torch.long)
    print(data.shape, data.dtype)

    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(len(train_data), len(val_data))

    model = BigramLanguageModel(vocab_size, block_size, device)
    m = model.to(device)

    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter == 5000:
            learning_rate *= 0.1

        if iter % 200 == 0:
            print(loss.item())

    @torch.no_grad()
    def estimate_loss():
        out = {}
        m.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = m(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        m.train()
        return out

    print('final loss', estimate_loss())