import random
from value_exp_div_pow import Value
import matplotlib.pyplot as plt

class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((xi * wi for xi, wi in zip(x, self.w)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

def mse_loss(y, ys):
    return (y - ys) ** 2

if __name__ == "__main__":
    n = MLP(3, [5, 5, 3])
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [
        [1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ]

    loss_history = []
    iteration = 100

    for k in range(iteration):

        ypred = [n(x) for x in xs]
        loss = 0
        # loss calculate (like PyTorch MSELoss(reduction="mean"))
        total_output = 0
        for a, b in zip(ypred, ys):
            loss += sum(mse_loss(c, d) for c, d in zip(a, b))
            total_output += len(a)
        loss /= total_output

        for p in n.parameters():
            p.grad = 0.0
        loss.backward()

        for key, p in enumerate(n.parameters()):
            p.data += -0.5 * p.grad
            # if key == 5:
            #     print(f"param {key}: data: {p.data}, grad: {p.grad}")

        # print("loss", loss)
        # print("w", n.layers[0].neurons[0].w[0])
        # print("w grad", n.layers[0].neurons[0].w[0].grad)

        # n.layers[0].neurons[0].w[0].data = n.layers[0].neurons[0].w[0].data - (0.1 * n.layers[0].neurons[0].w[0].grad)
        loss_history.append(loss.data)
        print(k, loss.data)

    plt.plot([i for i in range(iteration)], loss_history)
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.show()