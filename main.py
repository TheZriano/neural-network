import json
import math
import random
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Carica il dataset MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

print(f"Numero di immagini di training: {train_images.shape[0]}")
print(f"Dimensione di una singola immagine: {train_images.shape[1:]}")


class Neuron:
    def __init__(self, column, weights=0, value=0, bias=0):
        self.column=column
        self.weights=weights
        self.bias=bias
        self.value=value
        self.delta=0

    def calc(self):
        if self.column==0:
            return self.value
        else:
            previous_neurons = neuralNetwork[self.column - 1]
            weighted_sum = sum(neuron.calc() * weight for neuron, weight in zip(previous_neurons, self.weights))
            return function(weighted_sum+self.bias)
        
    def get_data(self):
        return {
            'weights': self.weights.tolist() if isinstance(self.weights, np.ndarray) else self.weights,
            'bias': self.bias,
            'value': self.value.tolist() if isinstance(self.value, np.ndarray) else self.value
        }

def saveNetworkToJson(path):
    network_data = []
    for column in neuralNetwork:
        column_data = []
        for neuron in column:
            column_data.append(neuron.get_data())
        network_data.append(column_data)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(network_data, file, indent=4)

def createNetworkFromJson(path):
    global neuralNetwork
    neuralNetwork = []
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    for columnIndex, column in enumerate(data):
        array = []
        for neuron in column:
            
            weights = neuron.get("weights", [random.uniform(-1, 1) for _ in range(len(data[columnIndex - 1]))]) if columnIndex > 0 else [0] * len(neuron)
            array.append(
                Neuron(
                    columnIndex,
                    weights=weights,
                    value=neuron.get("value", 0)
                )
            )
        neuralNetwork.append(array)
    return neuralNetwork

def toPercent(values):
    values=np.array(values)
    return values / values.sum() * 100


def function(x):
    return 1 / (1 + np.exp(-x))


def derivative(x):
    return x * (1 - x)

def train(input_values, target_values, learning_rate=0.1):

    for i, value in enumerate(input_values):
        neuralNetwork[0][i].value = value

    for i, neuron in enumerate(neuralNetwork[-1]):
        output = neuron.calc()
        error = target_values[i] - output
        neuron.delta = error * derivative(output)

    for layer_index in reversed(range(1, len(neuralNetwork) - 1)):
        for i, neuron in enumerate(neuralNetwork[layer_index]):
            output = neuron.calc()
            # somma dei pesi * delta dei neuroni successivi
            downstream = neuralNetwork[layer_index + 1]
            error = sum(n.weights[i] * n.delta for n in downstream)
            neuron.delta = error * derivative(output)

    for layer_index in range(1, len(neuralNetwork)):
        for neuron in neuralNetwork[layer_index]:
            for j, prev_neuron in enumerate(neuralNetwork[layer_index - 1]):
                neuron.weights[j] += learning_rate * neuron.delta * prev_neuron.calc()
            neuron.bias += learning_rate * neuron.delta




neuralNetwork=createNetworkFromJson("data.json")

def one_hot(label):
    arr = [0] * 10
    arr[label] = 1
    return arr

#training

"""for i in tqdm(range(50000,60000)):
    input_data = [x / 255 for x in train_images[i].flatten()]      # normalizza
    target = one_hot(train_labels[i])                    # etichetta
    train(input_data, target)
    if i%10==0:
        saveNetworkToJson("data.json")
"""

def testResults():
    correct=0
    wrong=[]
    localtestImages=test_images
    localtestLabel=test_labels
    for i in tqdm(range(100)):
        
        test_input = [x / 255 for x in localtestImages[i].flatten()]
        predicted_output=localtestLabel[i]

        for j, v in enumerate(test_input):
            neuralNetwork[0][j].value = v

        output = [neuron.calc() for neuron in neuralNetwork[-1]]
        if output.index(max(output))==predicted_output:
            correct+=1
        else:
            wrong.append(i)
    print(f"corretti) {correct}\n wrong) {wrong}")





test_input = [x / 255 for x in test_images[int(input("immagine? "))].flatten()]


for i, v in enumerate(test_input):
    neuralNetwork[0][i].value = v

output = [neuron.calc() for neuron in neuralNetwork[-1]]
output = toPercent(output)

for i, result in enumerate(output):
    print(f"{i}) {result:.2f}%")


# Converte la lista 1D in una matrice 2D (28x28 per MNIST)
input_data_2d = np.array(test_input).reshape(28, 28)

# Visualizza l'immagine
plt.imshow(input_data_2d, cmap='gray')  # 'gray' per immagini in bianco e nero
plt.axis('off')  # Disattiva gli assi
plt.show()

#testResults()