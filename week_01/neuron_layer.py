from forward_pass import Neuron

class NeuronLayer:
    def __init__(self, neurons_num):
        self.neurons = [Neuron() for _ in range(neurons_num)]

    def forward(self, inputs, weights, biases):
        output = []
        for neuron, w, bias in zip(self.neurons, weights, biases):
            z = neuron.forward(inputs, w, bias)
            output.append(z)
        return output

if __name__ == "__main__":
    inputs = [0.55, 0.13, 0.26, 0.33, 0.55, 0.67, 0.96, 0.22, 0.78, 0.37]
    weights = [
        [0.12, -0.34, 0.56, 0.21, -0.45, 0.67, 0.11, -0.23, 0.38, 0.52],
        [-0.41, 0.25, 0.73, -0.16, 0.34, -0.52, 0.19, 0.61, -0.28, 0.44],
        [0.35, -0.27, 0.48, 0.62, -0.13, 0.55, -0.39, 0.17, 0.71, -0.46],
        [-0.22, 0.64, -0.31, 0.45, 0.76, -0.18, 0.29, -0.57, 0.33, 0.51],
        [0.58, -0.12, 0.37, -0.49, 0.63, 0.24, -0.35, 0.72, -0.26, 0.41]
    ]
    biases = [ 0.10, -0.20, 0.05, 0.30, -0.15]
    neurons = NeuronLayer(neurons_num=5)
    output = neurons.forward(inputs, weights, biases)
    print(output)


