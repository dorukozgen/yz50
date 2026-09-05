import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

if __name__ == "__main__":

    # Task 1

    words = open('./hafta-3/names.txt', 'r').read().splitlines()

    t = {}

    for w in words:
        chs = ['.', '.'] + list(w) + ['.']
        for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
            trigram = (ch1, ch2, ch3)
            t[trigram] = t.get(trigram, 0) + 1

    print(sorted(t.items(), key = lambda kv : -kv[1]))

    chars = sorted(list(set(''.join(words))))
    stoi = { s:i+1 for i,s in enumerate(chars) }
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}

    print(itos)

    N = torch.zeros((27, 27, 27), dtype=torch.int32)

    # for i in range(27):
    #     for j in range(27):
    #         bigram = (itos[i], itos[j])
    #         N[i, j] = b.get(bigram, 0)

    for w in words:
        chs = ['.'] + ['.'] + list(w) + ['.']
        for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            ix3 = stoi[ch3]
            N[ix1, ix2, ix3] += 1

    print(N)

    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(projection="3d")
    for i in range(27):
        for j in range(27):
            for k in range(27):
                chstr = itos[i] + itos[j] + itos[k]
                ax.text(i, j, k, chstr, ha="center", va="bottom", color="gray")
                ax.text(i, j, k, N[i, j, k].item(), ha="center", va="top", color="gray")
    ax.axis("off")
    plt.show()

    # # Task 2

    # P = (N+1).float()
    # P /= P.sum(1, keepdim=True)

    # g = torch.Generator().manual_seed(2147483647)
    # for i in range(5):
    #     ix = 0
    #     out = []
    #     while True:
    #         p = P[ix]
    #         ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    #         out.append(itos[ix])
    #         if ix == 0:
    #             break
    #     print(''.join(out))

    # # Task 3

    # log_likelihood = 0
    # n = 0
    # for w in ['doruk']:
    #     chs = ['.'] + list(w) + ['.']
    #     for ch1, ch2 in zip(chs, chs[1:]):
    #         ix1 = stoi[ch1]
    #         ix2 = stoi[ch2]
    #         prob = P[ix1, ix2]
    #         logprob = torch.log(prob)
    #         log_likelihood += logprob
    #         n += 1
    # print(f"{log_likelihood=}")
    # nll = -log_likelihood
    # print(f"{nll=}")
    # print(nll/n)

    # # Task 4

    # xs, ys = [], []
    # for w in words:
    #     chs = ['.'] + list(w) + ['.']
    #     for ch1, ch2 in zip(chs, chs[1:]):
    #         ix1 = stoi[ch1]
    #         ix2 = stoi[ch2]
    #         xs.append(ix1)
    #         ys.append(ix2)
    # xs = torch.tensor(xs)
    # ys = torch.tensor(ys)
    # num = xs.nelement()
    # print("el number of xs:", num)

    # g = torch.Generator().manual_seed(2147483647)

    # W = torch.randn((27, 27), generator=g, requires_grad=True)

    # for k in range(100):

    #     zenc = F.one_hot(xs, num_classes=27).float()
    #     logits = zenc @ W

    #     #softmax
    #     counts = logits.exp()
    #     probs = counts / counts.sum(1, keepdim=True)

    #     loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()

    #     W.grad = None
    #     loss.backward()

    #     W.data += -50 * W.grad

    #     print("loss:", loss.item())

    # g = torch.Generator().manual_seed(2147483647)

    # for i in range(30):
    #     out = []
    #     ix = 0
    #     while True:
    #         xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    #         # print(xenc)
    #         logits = xenc @ W
    #         counts = logits.exp()
    #         probs = counts / counts.sum(1, keepdim=True)
    #         # print(probs)
    #         ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
    #         out.append(itos[ix])
    #         if ix == 0:
    #             break
    #     print(''.join(out))