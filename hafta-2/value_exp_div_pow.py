import math
from graphviz import Digraph
import torch

class Neuron:
    def forward(self, input: Value, weight: Value, bias: Value):
        z = input * weight + bias
        out = z.tanh()
        return out

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supports int/float"
        out = Value(self.data ** other, (self, ), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other ** -1
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other
    
    def tanh(self):
        out = Value((math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1), (self, ), "tanh")

        def _backward():
            self.grad += (1 - (out.data ** 2)) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        out = Value(math.exp(self.data), (self, ), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
            
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
    
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n._op, label = n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

if __name__ == "__main__":

    h = 0.001

    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    b = Value(7.0, label='b')

    x1w1 = x1*w1; x1w1.label = 'x1*w1'
    x2w2 = x2*w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 + b; n.label = 'n'

    e = (2*n).exp(); e.label = "e"
    o = (e - 1) / (e + 1)

    o.label = 'o'
    o.backward()

    L1 = o.data

    dot = draw_dot(o)
    dot.render('./hafta-2/values-graph.gv', view=True)
    print(dot.source)

    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    b = Value(7.0, label='b')

    x1w1 = x1*w1; x1w1.label = 'x1*w1'
    x2w2 = x2*w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = (x1w1 + h) + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 + b; n.label = 'n'

    e = (2*n).exp(); e.label = "e"
    o = (e - 1) / (e + 1)

    L2 = o.data

    print("numerical deriative x1w1", (L2 - L1) / h)

    x1 = torch.tensor([2.0], requires_grad=True)
    x2 = torch.tensor([0.0], requires_grad=True)

    w1 = torch.tensor([-3.0], requires_grad=True)
    w2 = torch.tensor([1.0], requires_grad=True)

    b = torch.tensor([7.0], requires_grad=True)

    x1w1 = x1 * w1; x1w1.retain_grad()
    x2w2 = x2 * w2; x2w2.retain_grad()
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.retain_grad()
    n = x1w1x2w2 + b; n.retain_grad()
    e = (2*n).exp(); e.retain_grad()
    o = (e - 1) / (e + 1); o.retain_grad()
    o.backward()

    print(f"toch grad x1w1: {x1w1.grad}")
    print(f"torch grad x2w2: {x2w2.grad}")
    print(f"torch grad x1w1x2w2: {x1w1x2w2.grad}")
    print(f"torch grad n: {n.grad}")
    print(f"torch grad e: {e.grad}")
    print(f"torch grad o: {o.grad}")


