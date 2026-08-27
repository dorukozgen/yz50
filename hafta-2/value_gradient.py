import math

class Neuron:
    def forward(self, input: Value, weight: Value, bias: Value):
        z = input * weight + bias
        out = z.tanh()
        return out

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        return out

    def tanh(self):
        out = Value((math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1), (self, ), "tanh")
        return out


if __name__ == "__main__":

    h = 0.001

    a = Value(2.0)
    b = Value(4.0)
    c = Value(-5.0)
    d = a + b
    e = d * c
    f = Value(6.0)
    L = e * f

    L1 = L.data
    
    a = Value(2.0)
    b = Value(4.0)
    c = Value(-5.0 + h)
    d = a + b
    e = d * c
    f = Value(6.0)
    L = e * f

    L2 = L.data

    print((L2 - L1) / h)

    L.grad = 1

    # dL / df = (d * (f+h)) - (e * f) / h = (e*f) + (e*h) - (e*f) / h = e = d * c = (-30)
    f.grad = -30.0

    # dL / de = (dL / de) = f = 6
    e.grad = 6.0

    # dL / dd = (dL / de) * (de / dd) = 6 * (-5) = (-30)
    d.grad = -30.0

    # dL / dc = (dL / de) * (de / dc) = 6 * 6 = 36
    c.grad = 36.0

    # dL / db = (dL / de) * (de / dd) * (dd * db)  = 6 * (-5) * 1 = (-30)
    b.grad = -30.0

    # dL / da = (dL / de) * (de / dd) * (dd * da)  = 6 * (-5) * 1 = (-30)
    a.grad = -30.0

    # neuron example
    x1 = Value(4.0)
    x2 = Value(1.0)

    w1 = Value(-1.0)
    w2 = Value(2.0)

    b = Value(1.0)

    x1w1 = x1 * w1
    x2w2 = x2 * w2
    x1w1x2w2 = x1w1 + x2w2
    n = x1w1x2w2 + b
    o = n.tanh()

    o.grad = 1

    # do / dn = 1 - (tanh(n) ** 2)
    n.grad = 1 - (o.data ** 2)

    x1w1x2w2.grad = n.grad
    b.grad = n.grad
    x2w2.grad = n.grad
    x1w1.grad = n.grad
    x2.grad = w2.data * x2w2.grad
    w2.grad = x2.data * x2w2.grad
    x1.grad = w1.data * x1w1.grad
    w1.grad = x1.data * x1w1.grad

    print(n.grad)









