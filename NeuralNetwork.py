import numpy as np

class NeuralNetwork:
    def __init__(self, learning_rate):
        # Inizializzazione dei pesi con valori casuali e del bias
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def compute_derivate(self, prediction, target):
        # Calcola la derivata dell'errore quadratico rispetto alla previsione
        return 2 * (prediction - target)

    def sigmoid(self, x):
        # Funzione di attivazione sigmoidale
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivate(self, x):
        # Derivata della funzione sigmoidale
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def make_prediction(self, input_vector):
        # Calcola l'output della rete neurale per un dato input
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self.sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def compute_gradient(self, input_vector, target):
        # Calcola il gradiente dell'errore rispetto ai pesi e al bias
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self.sigmoid(layer_1)
        prediction = layer_2

        # Derivata dell'errore rispetto alla previsione
        derror_dprediction = self.compute_derivate(prediction, target)
        # Derivata della previsione rispetto all'input del livello 1
        dprediction_dlayer1 = self.sigmoid_derivate(layer_1)
        # Derivata del livello 1 rispetto al bias
        dlayer1_dbias = 1
        # Derivata del livello 1 rispetto ai pesi
        dlayer1_dweights = input_vector

        # Derivata dell'errore rispetto al bias
        derror_dbias = derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        # Derivata dell'errore rispetto ai pesi
        derror_dweights = derror_dprediction * dprediction_dlayer1 * dlayer1_dweights

        return derror_dbias, derror_dweights

    def update_parameters(self, derror_dbias, derror_dweights):
        # Aggiorna i pesi e il bias della rete neurale
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (derror_dweights * self.learning_rate)

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Seleziona un esempio casuale dal set di dati di input
            random_data_index = np.random.randint(len(input_vectors))
            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Calcola i gradienti e aggiorna i pesi
            derror_dbias, derror_dweights = self.compute_gradient(input_vector, target)
            self.update_parameters(derror_dbias, derror_dweights)

            # Misura l'errore cumulativo per tutti gli esempi ogni 100 iterazioni
            if current_iteration % 100 == 0:
                cumulative_error = 0
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.make_prediction(data_point)
                    error = self.compute_derivate(prediction, target)
                    cumulative_error += error
                cumulative_errors.append(cumulative_error)
        return cumulative_errors


import matplotlib.pyplot as plt
input_vectors = np.array([[3, 1.5],[2, 1],[4, 1.5],[3, 4],[3.5, 0.5],[2, 0.5],[5.5, 1],[1, 1],])
targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])
learning_rate = 0.1
neural_network = NeuralNetwork(learning_rate)
training_error = neural_network.train(input_vectors, targets, 10000)
 
plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("cumulative_error.png")
plt.show()