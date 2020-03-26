import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def relu(x):
    return np.maximum(x, 0.)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logic_func(a, b, c):
    return (a and not b) or (c != b)

def element_wise_predict(dataset, weights):
    dataset = dataset.copy()
    activation = [relu for _ in range(len(weights) - 1)]
    activation.append(sigmoid)
    for w in range(len(weights)):
        res = np.zeros((len(dataset), len(weights[w][1])))
        for i in range(len(dataset)):
            for j in range(len(weights[w][1])):
                sum = 0
                for k in range(len(dataset[i])):
                    sum += dataset[i][k] * weights[w][0][k][j]
                res[i][j] = activation[w](sum + weights[w][1][j])
        dataset = res
    return dataset

def tensor_predict(dataset, weights):
    dataset = dataset.copy()
    activation = [relu for _ in range(len(weights) - 1)]
    activation.append(sigmoid)
    for i in range(len(weights)):
        dataset = activation[i](np.dot(dataset, weights[i][0]) + weights[i][1])
    return dataset

def prints(model, dataset):
    weights = []
    for layer in model.layers:
        weights.append(layer.get_weights())
    element_wise_res = element_wise_predict(dataset, weights)
    numpy_res = tensor_predict(dataset, weights)
    model_res = model.predict(dataset)
    print("Результат поэлементного вычисления:\n", element_wise_res)
    print("Результат тензорного вычисления:\n", numpy_res)
    print("Результат прогона через обученную модель:\n", model_res)

dataset = np.array([[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1]])
train_target = np.array([int(logic_func(x[0], x[1], x[2])) for x in dataset])

model = Sequential()
model.add(Dense(8, activation='relu', input_dim=3))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Корректный результат:\n", train_target)
prints(model, dataset)
model.fit(dataset, train_target, epochs=150, batch_size=1)
prints(model, dataset)
print("Корректный результат:\n", train_target)