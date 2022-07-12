# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import util


class ModelMaker:
    def __init__(
        self,
        samplerate,
        info_file,
        est_file,
        lr,
        data_size,
        batch_size,
        epochs,
        valid_rate,
    ):
        self.samplerate = samplerate
        self.info_file = info_file
        self.est_file = est_file
        self.lr = lr
        self.data_size = data_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.valid_rate = valid_rate

    def define_model(self, input_shape):
        model = Sequential()  # MLP model

        # 隠れ層1
        model.add(Dense(256, activation="relu", input_dim=input_shape))
        model.add(Dropout(0.25))
        # 隠れ層2
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.25))
        # 出力層
        model.add(Dense(10, activation="softmax"))

        # 学習過程の設定 -目的関数:categorical_crossentropy -最適化アルゴリズム:Adam
        model.compile(
            optimizer=Adam(lr=self.lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def fit_model(self):
        # load dataset
        train = pd.read_csv("../training.csv")  # [2700 rows x 2 columns]
        wav_train, label_train, duration_train = util.load_data(
            train, self.data_size, self.samplerate
        )

        # print("duration of training data")
        # print("max", max(duration_train))
        # print("min", min(duration_train))
        # print("mean", np.median(duration_train))

        # feature extraction
        if os.path.isfile("keras_model/my_mfcc.csv"):
            mfcc_train = pd.read_csv("keras_model/my_mfcc.csv", header=None).values
            print("read_csvしたよ", mfcc_train.shape)

        else:
            mfcc_train = util.feature_mfcc(wav_train)
            print("read_csvしてないよ", mfcc_train.shape)  # (2700, 26)

        # split into training data and validation data
        x_train, x_validation, lab_train, lab_validation = train_test_split(
            mfcc_train,
            label_train,
            test_size=self.valid_rate,
            random_state=3103,
        )

        # define model
        model = self.define_model(x_train.shape[1])

        # fit model
        early_stopping = EarlyStopping(monitor="val_loss", patience=2)

        # callback あり
        # history = model.fit(
        #     x_train, lab_train, batch_size=self.batch_size, shuffle=True,
        #     epochs=self.epochs, callbacks=[early_stopping], validation_data=(x_validation, lab_validation))

        # callback なし
        history = model.fit(
            x_train,
            lab_train,
            batch_size=self.batch_size,
            shuffle=True,
            epochs=self.epochs,
            validation_data=(x_validation, lab_validation),
        )

        # save fitted model
        model.save(self.est_file)
        score = model.evaluate(x_validation, lab_validation, verbose=0)
        print("test xentropy:", score)

        return model, history.history

    def execute(self):

        model, history = self.fit_model()
        # save network structure
        with open(self.info_file, "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))
        # plot_model(model, to_file = self.graph_file, show_shapes=True)

        # save training history
        if "acc" in history:
            history["accuracy"] = history.pop("acc")
            history["val_accuracy"] = history.pop("val_acc")
        util.plot(history, self.epochs)
        print("val_loss: %f" % history["val_loss"][-1])

        # evaluate with test data (accuracy & cm)
        if os.path.isfile("../test_truth.csv"):
            print("test file あり")
            test = pd.read_csv("../test_truth.csv")  # [300 rows x 1 columns]
            wav_test, _, _ = util.load_data(test, self.data_size, self.samplerate)
            label_test = test["label"]
            mfcc_test = util.feature_mfcc(wav_test)

            predict = model.predict(mfcc_test)
            predicted_values = np.argmax(predict, axis=1)
            print("test value", predicted_values)

            util.plot_confusion_matrix(predicted_values, label_test)
