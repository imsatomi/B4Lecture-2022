# -*- coding: utf-8 -*-
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data(data, data_size, samplerate):
    """
    Args:
        data(ndarray): data
        data_size(int): data size
        samplerate(int): sample rate
    Returns:
        wav_train(ndarray): wav of train data
        label_train(ndarray): label of train data
        duration(ndarray): duration
    """
    duration = np.zeros(len(data))
    wav_train = np.zeros((len(data), data_size))
    label_train = to_categorical(data["label"])  # one-hot

    for i, row in data.iterrows():
        y = librosa.load(f"../{row[0]}", sr=samplerate)[0]
        duration[i] = librosa.get_duration(y=y)  # 秒数取得->箱ひげ図
        # zero padding
        if len(y) > data_size:
            wav_train[i] = y[0:data_size]
        else:
            wav_train[i, 0 : len(y)] = y

    return wav_train, label_train, duration


def preEmphasis(signal, p=0.97):  # MFCCの前処理
    """pre emphasis filter
    Args:
        signal(ndarray): wav data
        p(float): p
    Returns:
        filtered wav data
    """
    return sp.signal.lfilter([1.0, -p], 1, signal)


def feature_mfcc(wav, n_mfcc=13):
    """mfcc
    Args:
        wav(ndarray): wav data
        n_mfcc(int): mfcc dimention
    Returns:
        features(ndarray): feature values
    """

    pre_wav = [preEmphasis(y) for y in wav]

    mfcc = np.array(
        list(map(lambda x: librosa.feature.mfcc(y=x, sr=8000, n_mfcc=n_mfcc), pre_wav))
    )
    ave_mfcc = np.array([np.mean(m, axis=1) for m in mfcc])

    # print("mfcc", ave_mfcc.shape)
    delta = np.array(list(map(lambda x: librosa.feature.delta(x), mfcc)))
    ave_delta = np.array([np.mean(m, axis=1) for m in delta])
    # print("delta", ave_delta.shape)
    features = np.concatenate((ave_mfcc, ave_delta), axis=1)
    # print("features", features.shape)

    # 保存
    if os.path.isfile("keras_model/my_mfcc.csv") == False:
        df = pd.DataFrame(features)
        df.to_csv("keras_model/my_mfcc.csv", index=False, header=False)

    return features


def plot(history, epoch):
    """plot learning process
    Args:
        history(ndarray): predicted values
        epoch(float): epoch
    Returns:
        Nothing
    """

    # 学習過程の可視化
    print("len", len(history["accuracy"]))
    epoch = len(history["accuracy"])
    plt.plot(range(1, epoch + 1), history["accuracy"], label="training", linestyle="--")
    plt.plot(range(1, epoch + 1), history["val_accuracy"], label="validation")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("figs/process_accuracy.png")
    plt.close()

    plt.plot(range(1, epoch + 1), history["loss"], label="training", linestyle="--")
    plt.plot(range(1, epoch + 1), history["val_loss"], label="validation")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("figs/process_loss.png")
    plt.close()


def plot_confusion_matrix(predict, ground_truth):
    """plot confusion matrix
    Args:
        predict(ndarray): predicted values
        ground_truth(ndarray): answer label
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
