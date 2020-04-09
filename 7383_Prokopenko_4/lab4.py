import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import optimizers


def load_img(path):
    img = load_img(path=path, target_size=(28, 28))
    return img_to_array(img)


mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images / 255.0
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(Dense(10, activation='softmax'))


def run_research(optimizer):
    optimizerConf = optimizer.get_config()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=4, batch_size=128, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('test_acc:', test_acc)
    print('optimizer', optimizerConf)
    plt.title('Training and test accuracy')
    plt.plot(history.history['accuracy'], 'r', label='train')
    plt.plot(history.history['val_accuracy'], 'b', label='test')
    plt.legend()
    plt.savefig("%s_%s_%s_acc.png" % (optimizerConf["name"], optimizerConf["learning_rate"], test_acc), format='png')
    plt.clf()

    plt.title('Training and test loss')
    plt.plot(history.history['loss'], 'r', label='train')
    plt.plot(history.history['val_loss'], 'b', label='test')
    plt.savefig("%s_%s_%s_loss.png" % (optimizerConf["name"], optimizerConf["learning_rate"], test_acc), format='png')
    plt.legend()
    plt.clf()
    return model

for learning_rate in [0.001, 0.01]:
    run_research(optimizers.Adagrad(learning_rate=learning_rate))
    run_research(optimizers.Adam(learning_rate=learning_rate))
    run_research(optimizers.RMSprop(learning_rate=learning_rate))
    run_research(optimizers.SGD(learning_rate=learning_rate))
