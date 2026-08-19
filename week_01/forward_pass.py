import math

class Neuron:
    def forward(self, inputs, weights, bias):
        z = 0
        for input, weight in zip(inputs, weights):
            z += input * weight
        z += bias
        sigmoid = (1 / (1 + math.exp(-z)))
        return sigmoid



if __name__ == "__main__":
    inputs = [0.28, 0.56]
    weights = [0.12, -0.34]
    bias = 3
    neuron = Neuron()
    output = neuron.forward(inputs, weights, bias)
    print(output)
