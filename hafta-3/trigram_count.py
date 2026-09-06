import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

if __name__ == "__main__":

    # Task 1

    words = open('./hafta-3/names_dev.txt', 'r').read().splitlines()

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
        chs = ['.', '.'] + list(w) + ['.']
        for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            ix3 = stoi[ch3]
            N[ix1, ix2, ix3] += 1

    print(N)

    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111, projection="3d")
    x_idx, y_idx, z_idx = torch.where(N > 1)
    freqs = N[x_idx, y_idx, z_idx] 

    sizes = (freqs / freqs.max()) * 100 

    sc = ax.scatter(x_idx, y_idx, z_idx, c=freqs, s=sizes, cmap="viridis", alpha=0.7)

    ax.set_xticks(range(len(chars)))
    ax.set_xticklabels(chars, rotation=90)
    ax.set_yticks(range(len(chars)))
    ax.set_yticklabels(chars, rotation=0)
    ax.set_zticks(range(len(chars)))
    ax.set_zticklabels(chars, rotation=0)

    ax.set_xlabel("First Character")
    ax.set_ylabel("Second Character")
    ax.set_zlabel("Third Character")
    ax.set_title("3D Visualization of Trigram Frequencies")

    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label("Frequency")
    # plt.show()

    # Task 2

    P = (N+1).float()

    g = torch.Generator().manual_seed(2147483647)
    for i in range(10):
        ix1, ix2 = 0, 0
        out = []
        while True:
            p = P[ix1, ix2].float()
            p /= p.sum()
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            ix1 = ix2
            ix2 = ix
            if ix2 == 0:
                break
        print(''.join(out))

    # # Task 3

    log_likelihood = 0
    n = 0
    for w in (words + ["andrej"]):
        chs = ['.', '.'] + list(w) + ['.']
        for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            ix3 = stoi[ch3]
            p = P[ix1, ix2].float()
            p /= p.sum()
            logprob = torch.log(p[ix3])
            log_likelihood += logprob
            n += 1
    print(f"{log_likelihood=}")
    nll = -log_likelihood
    print(f"{nll=}")
    print(nll/n)

    # # Task 4

    xs, ys = [], []
    for w in words:
        chs = ['.', '.'] + list(w) + ['.']
        for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            ix3 = stoi[ch3]
            xs.append([ix1, ix2])
            ys.append(ix3)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    num = xs.nelement()
    print("el number of xs:", num)

    g = torch.Generator().manual_seed(2147483647)

    xenc = F.one_hot(xs, num_classes=27).float()
    num_sequences = xs.size(0)
    xenc_flat = xenc.view(num_sequences, -1)
    W = torch.randn((54, 27), generator=g, requires_grad=True)
    logits = xenc_flat @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)

    neg_log_liklihood = torch.zeros(5)
    for i in range(5):
        x1, x2 = xs[i].tolist()
        y = ys[i].item()

        print(f'trigram example {i+1}: {itos[x1]}{itos[x2]}{itos[y]} (indexes {x1},{x2},{y})')
        print(f'input to the neural net:', x1, x2)
        print(f'output probablities from the neural net', probs[i])
        print(f'lable (actual next character)', y)
        p = probs[i, y] # Probablity of the correct next charater
        print(f'probablity assigned by the neural net to the correct charater', p.item())
        logp = torch.log(p) # log probablity
        print(f'log liklihood', logp.item())
        neg_log_liklihood[i] = -logp
        print(f'negative log liklihood:', neg_log_liklihood[i].item())
        #neg_log_liklihood[i] = neg_log_liklihood

    print('=========')
    print('average negative log liklihood:', neg_log_liklihood.mean().item())

    for k in range(100):

        xenc = F.one_hot(xs, num_classes=27).float()
        num_sequences = xs.size(0)
        xenc_flat = xenc.view(num_sequences, -1)
        logits = xenc_flat @ W

        #softmax
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)

        loss = -probs[torch.arange(num_sequences), ys].log().mean() + 0.01*(W**2).mean()

        W.grad = None
        loss.backward()

        with torch.no_grad():
            W += -30 * W.grad 

        print("loss:", loss.item())

    g = torch.Generator().manual_seed(2147483647)

    for i in range(100):
        out = []
        ix1, ix2 = 0, 0
        while True:
            xenc1 = F.one_hot(torch.tensor([ix1]), num_classes=27).float()
            xenc2 = F.one_hot(torch.tensor([ix2]), num_classes=27).float()
            xenc = torch.cat((xenc1, xenc2), dim=1).view(1, -1)
            # print(xenc)
            logits = xenc @ W
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdim=True)
            # print(probs)
            ix1 = ix2
            ix2 = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
            # print(ix2)
            out.append(itos[ix2])
            if ix2 == 0:
                break
        print(''.join(out))