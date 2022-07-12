# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import util


class ModelMaker:

    def __init__(self, samplerate, info_file, graph_file, est_file, lr, data_size,batch_size,epochs,valid_rate):
        self.samplerate = samplerate
        self.info_file = info_file
        self.graph_file = graph_file
        self.est_file = est_file
        self.lr = lr
        self.data_size = data_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.valid_rate = valid_rate

    # モデルを定義するメソッド
    def define_model(self, input_shape):
        model = Sequential()
        # MLP model
        # 隠れ層1  -ノード数:256 -活性化関数:relu -入力:X.shape[1:]次元 -ドロップアウト比率0.25
        # 最初の層では想定する入力データshapeを入力する必要あり
        print("X.shape", input_shape)  # X.shape (2700, 30, 2, 9)

        model.add(Dense(256, activation="relu", input_shape=input_shape))
        model.add(Dropout(0.25))
        # 隠れ層2  -ノード数:256 -活性化関数:relu -ドロップアウト比率0.25
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.25))
        # 出力層 -ノード数:10 -活性化関数:softmax
        model.add(Dense(10, activation="softmax"))

        # 学習過程の設定 -目的関数:categorical_crossentropy -最適化アルゴリズム:Adam
        model.compile(optimizer=Adam(lr=self.lr), loss="categorical_crossentropy",
                    metrics=["accuracy"])

        return model

    # モデルを訓練するメソッド
    def fit_model(self):
        # データセットの読み込み
        train = pd.read_csv("../training.csv")  # [2700 rows x 2 columns]
        wav_train, label_train, duration_train = util.load_data(train, self.data_size, self.samplerate)
        
        # data内容表示
        print("duration of training data")
        print("max", max(duration_train))
        print("min", min(duration_train))
        print("mean", np.median(duration_train))

        # 特徴量抽出
        if os.path.isfile("keras_model/my_mfcc.csv"):
            mfcc_train = pd.read_csv("keras_model/my_mfcc.csv").values
            mfcc_train = mfcc_train.reshape(2700, 60, 9)
            print("read_csvしたよ", mfcc_train.shape)
        else:
            mfcc_train = util.feature_mfcc(wav_train)
            print("read_csvしてないよ",mfcc_train.shape)  # (2700, 60, 9)

        # mfcc.shape[1:] (60, 9)
        
        # 学習データを学習データとバリデーションデータに分割 (バリデーションセットを20%とした例)
        x_train, x_validation, lab_train, lab_validation = train_test_split(
        mfcc_train, label_train,
        test_size=0.2,
        random_state=3103,
        )
        print("x_train.shape",x_train.shape)
        print("x_validation.shape",x_validation.shape)
        print("lab_train.shape",lab_train.shape)
        print("lab_validation.shape",lab_validation.shape)
        # x_train.shape (2160, 60, 9)
        # x_validation.shape (540, 60, 9)
        # lab_train.shape (2160, 10)
        # lab_validation.shape (540, 10)
        # ValueError: Shapes (None, 10) and (None, 60, 10) are incompatible

        # 次元調整
        # mfcc_train = np.reshape(mfcc_train, (2700, 30, 2, 9))
        # x_train = x_train[:, :, :, np.newaxis]
        print("次元調整", x_train.shape)

        # モデルを定義
        model = self.define_model(x_train.shape[1:])

        # (データ数, データ種類数, 説明変数)

        # 訓練の実行
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        history = model.fit(
            x_train, lab_train, batch_size=self.batch_size, shuffle=True,
            epochs=self.epochs, callbacks=[early_stopping], validation_data=(x_validation, lab_validation))

        # 訓練したモデルを保存
        model.save(self.est_file)
        score = model.evaluate(x_validation, lab_validation, verbose=0)
        print('test xentropy:', score)

        return model, history.history

    def execute(self):
        
        model, history = self.fit_model()
        # ネットワーク構造を保存
        with open(self.info_file, "w") as f:
            model.summary(print_fn = lambda x: f.write(x + "\n"))
        # plot_model(model, to_file = self.graph_file, show_shapes=True)

        # 訓練状況を保存
        if "acc" in history:
            history["accuracy"] = history.pop("acc")
            history["val_accuracy"] = history.pop("val_acc")
        util.plot(history, self.epochs)
        print("val_loss: %f" % history["val_loss"][-1])


        # テストデータで評価を行う（accuracyと混同行列）
        if os.path.isfile("../test_truth.csv"):
            print("test file あり")
            test = pd.read_csv("../test_truth.csv")  # [300 rows x 1 columns]
            wav_test, label_test, _= util.load_data(test, self.data_size, self.samplerate)

            predict = model.predict(wav_test)
            predicted_values = np.argmax(predict, axis=1)
            
            util.plot_confusion_matrix(predicted_values, label_test)




