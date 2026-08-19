from forward_pass import Neuron
from loss_function import loss
        
if __name__ == "__main__":
    learning_rate = 0.05
    h = 0.0001
    input = 0.55
    weight = -0.66
    bias = 1
    actual = 1
    neuron = Neuron()
    for i in range(50):
        d1 = neuron.forward([input], [weight], bias)
        d2 = neuron.forward([input], [weight+h], bias)
        l1 = loss(d1, actual)
        l2 = loss(d2, actual)
        slope = ((l2 - l1) / h)
        weight = weight - (learning_rate * slope)
        print("d1", d1)
        print("d2", d2)
        print("loss1", l1)
        print("loss2,", l2)
        print("slope", ((l2 - l1) / h))
    print("w_min", weight)
    
