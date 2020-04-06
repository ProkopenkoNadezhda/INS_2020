import numpy as np
import csv

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

X_MIN = -5
X_MAX = 10
E_MIN = 0
E_MAX = 0.3
TRAIN_SIZE = 100
TEST_SIZE = 50

def getData(size):
    dataset = []
    dataset_y = []
    for i in range(size):
        X = np.random.normal(X_MIN, X_MAX)
        e = np.random.normal(E_MIN, E_MAX)
        x = []
        y = []
        x.append(np.round((-X)**3 + e, decimals=3))
        x.append(np.round(np.log(np.fabs(X)) + e, decimals=3))
        x.append(np.round(np.sin(3*X) + e, decimals=3))
        x.append(np.round(np.exp(X) + e, decimals=3))
        x.append(np.round(X + 4 + e,decimals=3))
        x.append(np.round(-X + np.sqrt(np.fabs(X)) + e, decimals=3))
        y.append(np.round(X + e, decimals=3))
        dataset.append(x)
        dataset_y.append(y)
    return np.round(np.array(dataset), decimals=3), np.round(np.array(dataset_y), decimals=3)

def create():
    main_input = Input(shape=(6,), name='mainInput')

    enc = Dense(64, activation='relu')(main_input)
    enc = Dense(32, activation='relu')(enc)
    enc = Dense(6, activation='relu', name="encode")(enc)

    input2 = Input(shape=(6,), name='input_encoded')

    dec = Dense(32, activation='relu')(input2)
    dec = Dense(64, activation='relu')(dec)
    dec = Dense(6, name='decode')(dec)

    pred = Dense(64, activation='relu')(enc)
    pred = Dense(32, activation='relu')(pred)
    pred = Dense(16, activation='relu')(pred)
    pred = Dense(1, name="predict")(pred)

    enc = Model(main_input, enc, name="encoder")
    dec = Model(input2, dec, name="decoder")
    pred = Model(main_input, pred, name="autoencoder")
    return enc, dec, pred, main_input


def write_csv(path, data):
    with open(path, 'w', newline='') as file:
        output = csv.writer(file, delimiter=',')
        for i in data:
            output.writerow(np.round(i, decimals=3))


x_train, y_train = getData(TRAIN_SIZE)
x_test, y_test = getData(TEST_SIZE)

mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train -= mean
x_train /= std
x_test -= mean
x_test /= std

y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0)
y_train -= y_mean
y_train /= y_std
y_test -= y_mean
y_test /= y_std

encoder, decoder, autoEncoder, mainInput = create()

autoEncoder.compile(optimizer="adam", loss="mse", metrics=['mae'])
autoEncoder.fit(x_train, y_train, epochs=50, batch_size=1, validation_data=(x_test, y_test))

encoded_data = encoder.predict(x_test)
decoded_data = decoder.predict(encoded_data)
regr = autoEncoder.predict(x_test)

write_csv('./x_train.csv', x_train)
write_csv('./y_train.csv', y_train)
write_csv('./x_test.csv', x_test)
write_csv('./y_test.csv', y_test)
write_csv('./encoded.csv', encoded_data)
write_csv('./decoded.csv', decoded_data)
write_csv('./regression_predicted.csv', regr)

#save models
decoder.save('decoder.h5')
encoder.save('encoder.h5')
autoEncoder.save('regressor.h5')