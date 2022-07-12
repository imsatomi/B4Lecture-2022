# -*- coding: utf-8 -*-

import librosa
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import scipy as sp
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix


# 訓練用データセットを取得する関数
def load_data(data, DATA_SIZE, SAMPLERATE):
    duration = np.zeros(len(data))
    wav_train = np.zeros((len(data), DATA_SIZE))
    label_train = to_categorical(data["label"])  # one-hot

    for i, row in data.iterrows():
        y = librosa.load(f"../{row[0]}", sr=SAMPLERATE)[0]
        duration[i] = librosa.get_duration(y=y)  # 秒数取得->箱ひげ図
        # zero padding
        if (len(y) > DATA_SIZE):
            wav_train[i] = y[0:DATA_SIZE]
        else:
            wav_train[i, 0:len(y)] = y

    return wav_train, label_train, duration

def preEmphasis(signal, p=0.97):  # MFCCの前処理
    """プリエンファシスフィルタ"""
    # 係数 (1.0, -p) のFIRフィルタを作成
    return sp.signal.lfilter([1.0, -p], 1, signal)

def feature_mfcc(wav):
    
    pre_wav = [preEmphasis(y) for y in wav]

    mfcc = np.array(list(map(lambda x: librosa.feature.mfcc(y=x, sr=8000, n_mfcc=30), pre_wav)))
    # print("mfcc", mfcc.shape)  # mfcc (2700, 30, 9)
    delta = np.array(list(map(lambda x: librosa.feature.delta(x), mfcc)))
    # print("delta", delta.shape)  # delta (2700, 30, 9)
    features = np.concatenate((mfcc, delta), axis=1)
    # print("features", features.shape)  # features (2700, 60, 9)

    # 保存
    features = features.reshape(2700, -1)
    df = pd.DataFrame(features)
    df.to_csv("keras_model/my_mfcc.csv")
    features = features.reshape(2700, 60, 9)

    return features

def plot(history, EPOCH):

    # 学習過程の可視化
    plt.plot(range(1, EPOCH+1), history['accuracy'], label="training", linestyle = "--")
    plt.plot(range(1, EPOCH+1), history['val_accuracy'], label="validation")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("figs/" + "process_accuracy.png")
    plt.close()

    plt.plot(range(1, EPOCH+1), history['loss'], label="training", linestyle = "--")
    plt.plot(range(1, EPOCH+1), history['val_loss'], label="validation")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("figs/" + "process_loss.png")
    plt.close()

def plot_confusion_matrix(predict, ground_truth):
    """
    予測結果の混合行列をプロット
    Args:
        predict: 予測結果
        ground_truth: 正解ラベル
        title: グラフタイトル
    Returns:
        Nothing
    """
    cm = confusion_matrix(predict, ground_truth)
    acc = np.sum(ground_truth == predict) / ground_truth.shape[0] * 100
    print("Test accuracy: ", accuracy_score(ground_truth, predict))

    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap="binary")
    plt.title(f"Results\n Acc: {acc}%")
    plt.colorbar()
    plt.ylabel("Predicted")
    plt.xlabel("Ground truth")
    plt.tight_layout()
    plt.savefig("figs/cm.png")
    plt.close()