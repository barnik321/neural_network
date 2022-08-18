from network import Model
import numpy as np
# from sklearn.datasets import load_iris
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from network import common
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # load iris data
    # x, y = load_iris(return_X_y=True)
    # x = x / x.max(axis=0)
    # y = common.one_hot_encode(y, [0, 1, 2])
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    #
    # mlp = Model()
    # mlp.add_layer(4, 10, act_fn='sigmoid')
    # mlp.add_layer(10, 3, act_fn='softmax')
    # losses, accs = mlp.fit(x_train, y_train, epochs=20, batch_size=5, loss='crossentropy', learning_rate=0.1)

    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # train_data = np.loadtxt('train.csv', delimiter=',')
    # test_data = np.loadtxt('test.csv', delimiter=',')
    # y_train = train_data[:, -1]
    # x_train = train_data[:, :-1]
    # y_test = test_data[:, -1]
    # x_test = test_data[:, :-1]
    # del train_data
    # del test_data

    x_train = x_train.reshape(x_train.shape[0], -1) / 255
    y_train = common.one_hot_encode(y_train)
    x_test = x_test.reshape(x_test.shape[0], -1) / 255
    y_test = common.one_hot_encode(y_test)

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)

    mlp = Model()
    mlp.add_layer(784, 200, act_fn='sigmoid')
    mlp.add_layer(200, 128, act_fn='sigmoid')
    mlp.add_layer(128, 64, act_fn='sigmoid')
    mlp.add_layer(64, 10, act_fn='softmax')
    losses, accs = mlp.fit(x_train, y_train, validate_x=x_valid, validate_y=y_valid,
                           epochs=20, batch_size=256, loss='crossentropy', learning_rate=0.001)

    # =============================================================
    pred = mlp.predict(x_test)
    pred = common.convert_to_binary(pred)

    common.ConfusionMatrixDisplay.from_predictions(common.convert_to_numpy(common.convert_to_binary(y_test)),
                                                   common.convert_to_numpy(pred))
    plt.show()

    print('Accuracy: ', common.accuracy(common.convert_to_binary(y_test), pred))

    plt.plot(np.arange(len(losses)), losses, label='Loss')
    plt.plot(np.arange(len(accs)), accs, label='Accuracy')
    plt.legend()
    plt.show()
