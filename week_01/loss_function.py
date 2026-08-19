from forward_pass import Neuron 
import matplotlib.pyplot as plt

def loss(predict, actual):
    return ((predict - actual) ** 2)

if __name__ == "__main__":
    loss_results = []
    weights = []
    input = 0.33
    weight = -0.16
    bias = 2
    actual = 1
    neuron = Neuron()
    output = neuron.forward([input], [weight], bias)
    err = loss(output, actual)
    loss_results.append(err)
    weights.append(weight)
    weight = 0.22
    output = neuron.forward([input], [weight], bias)
    err = loss(output, actual)
    loss_results.append(err)
    weights.append(weight)
    plt.plot(weights, loss_results, label="loss graph")
    plt.xlabel("weight")
    plt.ylabel("loss")
    plt.show()



    