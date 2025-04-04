import json
import math
import random

class Neuron:
    def __init__(self, column, weights=0, value=0):
        self.column=column
        self.weights=weights
        self.bias=1
        self.value=value

    def calc(self):
        if self.column==0:
            return self.value
        else:
            previous_neurons = neuralNetwork[self.column - 1]
            weighted_sum = sum(neuron.calc() * weight for neuron, weight in zip(previous_neurons, self.weights))
            return function(weighted_sum+self.bias)
        


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

def softMax(values):
    exp_values = [math.exp(v) for v in values]
    total = sum(exp_values)
    return [v / total for v in exp_values]


def function(x):
    return 1 / (1 + math.exp(-x))


def derivative(x):
    return x * (1 - x)

def train(input_values, target_values, learning_rate=0.1):

    for i, value in enumerate(input_values):
        neuralNetwork[0][i].value = value


    outputs = [neuron.calc() for neuron in neuralNetwork[-1]]

    errors = [(target - output) for target, output in zip(target_values, outputs)]

    for i, neuron in enumerate(neuralNetwork[-1]):  # Per ogni neurone nel layer di output
        output = neuron.calc()
        delta = errors[i] * derivative(output)  # Calcola l'errore e la sua derivata
        for j, prev_neuron in enumerate(neuralNetwork[neuron.column - 1]):
            neuron.weights[j] += learning_rate * delta * prev_neuron.value  # Aggiorna i pesi

    # 5. Aggiorna il bias
    for i, neuron in enumerate(neuralNetwork[-1]):
        neuron.bias += learning_rate * errors[i]  # Aggiorna il bias per l'output
    









neuralNetwork=createNetworkFromJson("data.json")


for epoch in range(1000):
    # Addestra la rete con vari esempi
    train([1, 0], [1])  # Eseguiamo un esempio di training con input [1, 0] e target 1
    train([0, 1], [0])  # Eseguiamo un altro esempio di training
    train([1, 1], [0])  # Eseguiamo un altro esempio
    train([0, 0], [0])  # Eseguiamo un altro esempio




test_input = [1, 0]  # Un input per il test
train(test_input, [1])  # Allena la rete con questo input e il valore atteso

# Ottieni e stampa i risultati
output = [neuron.calc() for neuron in neuralNetwork[-1]]
output = softMax(output)  # Applica softmax per ottenere probabilità
for i, result in enumerate(output):
    print(f"Classe {i}: {result*100:.2f}%")



#da aggiungere piu livelli