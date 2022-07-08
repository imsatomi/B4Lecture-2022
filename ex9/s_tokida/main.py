# -*- coding: utf-8 -*-

import argparse
import time

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import ShuffleSplit, train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam  #本
from tensorflow.keras.utils import to_categorical


NUM_CLASSES = 10
SAMPLERATE = 8000
LEN_DATA = 4096
BATCH_SIZE = 128
EPOCH = 20


def feature_mfcc(wav):

    data = [sp.signal.lfilter([1.0, -0.97], 1, signal) for signal in wav]

    mfcc = np.array(list(map(lambda x: librosa.feature.mfcc(y=x, sr=8000, n_mfcc=30), data)))
    delta = np.array(list(map(lambda x: librosa.feature.delta(x), mfcc)))
    
    features = np.concatenate((mfcc, delta), axis=1)

    return features


def cross_validation(X, Y):

    kfold = ShuffleSplit(n_splits=3, test_size=0.1, random_state=0)
    cvscores = []

    for cv_count, (train, validation) in enumerate(kfold.split(X, Y)):

        model = Sequential()

        model.add(Dense(256, activation="relu", input_dim=X.shape[1]))
        model.add(Dropout(0.25))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(10, activation="softmax"))
        model.compile(optimizer=Adam(lr=1e-4), loss="categorical_crossentropy",
                      metrics=["accuracy"])
        model.fit(X[train], y=Y[train],
                  batch_size=40, epochs=300, verbose=0)

        scores = model.evaluate(X[validation],
                                y=Y[validation],
                                verbose=0)
        print("accuracy : {:.2f}% ({}th CV)".format(scores[1] * 100, cv_count+1))
        cvscores.append(scores[1] * 100)

    print("{:.2f}% (+/- {:.2f}%)".format(np.mean(cvscores), np.std(cvscores)))


def fit_model(X, Y):

    # create model
    model = Sequential()

    model.add(Dense(256, activation="relu", input_dim=X.shape[1]))
    model.add(Dropout(0.25))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer=Adam(lr=1e-4), loss="categorical_crossentropy",
                      metrics=["accuracy"])

    model.compile(optimizer=Adam(lr=1e-4), loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # split data to train and validation
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, Y, test_size=0.2, random_state=int(np.random.rand() * 1e7))

    modelCheckpoint = ModelCheckpoint(filepath='keras_model/my_model.h5',
                                      save_best_only=False)

    result = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1,
                       validation_data=(X_validation, Y_validation),
                       callbacks=[modelCheckpoint])

    save_model_info("figs/model_info.txt", model)

    plt.plot(range(1, EPOCH+1), result.history['accuracy'], label="training")
    plt.plot(range(1, EPOCH+1), result.history['val_accuracy'], label="validation")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("figs/" + "process_accuracy.png")
    plt.show()

    plt.plot(range(1, EPOCH+1), result.history['loss'], label="training")
    plt.plot(range(1, EPOCH+1), result.history['val_loss'], label="validation")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("figs/" + "process_loss.png")
    plt.show()


    model = load_model('keras_model/my_model.h5')
    model.summary()
    score = model.evaluate(X_validation, Y_validation, verbose=0)
    print("validation loss :", score[0])
    print("validation accuracy :", score[1])


def load_data(data):
    duration = np.zeros(len(data))
    wav_train = np.zeros((len(data), LEN_DATA))

    for i, row in data.iterrows():
        # print(row[0])  # dataset/train/jackson_0.wav
        y = librosa.load(f"../{row[0]}", sr=SAMPLERATE)[0]
        duration[i] = librosa.get_duration(y=y)  # duration -> box plot
        # zero padding
        if (len(y) > LEN_DATA):
            wav_train[i] = y[0:LEN_DATA]
        else:
            wav_train[i, 0:len(y)] = y

    return wav_train, duration

def save_model_info(info_file, model):
    with open(info_file, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))


def main():
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    start_time = time.time()

    # load file
    train = pd.read_csv("../training.csv")  # [2700 rows x 2 columns]
    wav_train, duration_train = load_data(train)
    label_train = to_categorical(train["label"])  # one-hot

    test = pd.read_csv("../test.csv")  # [300 rows x 1 columns]
    _, duration_test = load_data(test)
    # wav_test, duration_test = load_data(test)
    # label_test = to_categorical(test["label"])  # one-hot

    # data内容表示
    print("duration of training data")
    print("max", max(duration_train))
    print("min", min(duration_train))
    print("mean", np.median(duration_train))
    l_time = time.time()
    print("file loaded:", l_time - start_time)
    print("duration of test data")
    print("max", max(duration_test))
    print("min", min(duration_test))
    print("mean", np.median(duration_test))

    # fig, ax = plt.subplots()
    # ax.boxplot((duration_train, duration_test))
    # ax.set_xticklabels(["training", "test"])
    # plt.title("duration")
    # plt.grid()
    # plt.savefig("figs/" + "duration.png")
    
    # feature extraction
    X = feature_mfcc(wav_train)
    Y = label_train
    # print(X)
    # print(Y)
    f_time = time.time()
    print("feature_xtraction:", f_time - start_time)

    # cross validation
    cross_validation(X, Y)
    c_time = time.time()
    print("cross_validation",c_time- start_time)

    # fitting
    fit_model(X, Y)
    fit_time = time.time()
    print("fit_model",fit_time - start_time)

    # test





if __name__ == "__main__":
    main()