import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import Model, load_model
import scipy


def visualize(pixels):
    # visualize a pixel matrix
    plt.imshow(pixels.reshape([28, 28]), cmap='gray');
    plt.show();

def toVec(L):
    # convert labels to vertors
    Y = np.zeros((L.shape[0], 10));
    for i in range(L.shape[0]):
        Y[i, L[i, 0]] = 1;

    return Y;

def pre():
    # processes csv to ndarray&divides data
    input = pd.read_csv("train.csv");
    input = np.array(input);
    print("Inputshape" + str(input.shape));
    X = input[:, 1:];
    Y = input[:, 0:1];
    print("Xshape" + str(X.shape));
    print("Yshape" + str(Y.shape));
    X_train = X[:30000, :].reshape(30000, 784, 1);
    X_dev = X[30000:36000, :].reshape(6000, 784, 1);
    X_test = X[36000:, :].reshape(6000, 784, 1);
    Y_train = toVec(Y[:30000, :]);
    Y_dev = toVec(Y[30000:36000, :]);
    Y_test = toVec(Y[36000:, :]);
    print(str(X_train.shape) + str(Y_train.shape));
    print(str(X_dev.shape) + str(Y_dev.shape));
    print(str(X_test.shape) + str(Y_test.shape));
    np.save("X_train.npy", X_train);
    np.save("X_dev.npy", X_dev);
    np.save("X_test.npy", X_test);
    np.save("Y_train.npy", Y_train);
    np.save("Y_dev.npy", Y_dev);
    np.save("Y_test.npy", Y_test);

def submit(model, filename):
    input = pd.read_csv("test.csv");
    input = np.array(input);
    print("Inputshape" + str(input.shape));
    X = input.reshape(28000, 784, 1) / 255.0;
    Y_pred = predict(model, X);
    frame = DataFrame(Y_pred, columns=["Label"]);
    frame.index = np.arange(1, len(frame) + 1);
    frame.to_csv(filename);

def fastPre():
    # loads ndarrays from disk
    X_train = np.load("X_train.npy") / 255.0;
    X_dev = np.load("X_dev.npy") / 255.0;
    X_test = np.load("X_test.npy") / 255.0;
    Y_train = np.load("Y_train.npy");
    Y_dev = np.load("Y_dev.npy");
    Y_test = np.load("Y_test.npy");
    print("train: " + str(X_train.shape) + str(Y_train.shape));
    print("dev: " + str(X_dev.shape) + str(Y_dev.shape));
    print("test: " + str(X_test.shape) + str(Y_test.shape));
    print("#4: " + str(Y_train[3,:]));
    visualize(X_train[3, :, :]);
    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test

def model_1(inputShape):
    # keras model 1: FC-NN
    # devacc 96%
    X_input = Input(inputShape);
    X = Flatten()(X_input)
    X = Dense(30, activation='relu')(X);
    X = Dropout(0.2)(X);
    X = BatchNormalization(axis=-1)(X);
    X = Dense(30, activation='relu')(X);
    X = Dropout(0.3)(X);
    X = BatchNormalization(axis=-1)(X);
    X = Dense(10, activation='softmax')(X);
    model = Model(inputs=X_input, outputs=X);
    return model;

def model_2(inputShape):
    # keras model 2: CNN
    #devacc 98%
    X_input = Input(inputShape);
    X = Reshape((28, 28, 1), input_shape=(784, 1))(X_input);
    X = Conv2D(7, (5,5), strides=(1,1))(X);
    X = BatchNormalization(axis=-1)(X);
    X = Activation('tanh')(X);
    X = MaxPool2D((3,3))(X);
    X = Flatten()(X);
    X = Dropout(0.2)(X);
    X = Dense(30, activation='relu')(X);
    X = Dropout(0.3)(X);
    X = BatchNormalization(axis=-1)(X);
    X = Dense(10, activation='softmax')(X);
    model = Model(inputs=X_input, outputs=X);
    return model;

def runModel(numEpochs, model):
    # run model
    X_train, X_dev, X_test, Y_train, Y_dev, Y_test = fastPre();
    model.summary();
    model.compile(optimizer='Adam', loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x=X_train, y=Y_train, epochs=numEpochs, batch_size=32);
    devLoss, devAcc = model.evaluate(x=X_dev, y=Y_dev)
    print("devLoss: " + str(devLoss));
    print("devAcc: " + str(devAcc));

def predict(model, X):
    # make predictions
    Y_pred = np.argmax(model.predict(X), axis=1);
    return Y_pred;





