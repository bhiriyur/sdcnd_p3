from keras.models import Sequential, load_model
from keras.layers import Convolution2D, Dense, MaxPooling2D, Dropout, BatchNormalization, Flatten, Input
from keras.preprocessing.image import img_to_array, load_img
from keras import backend as K
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def read_data(plot_hist=False):
    """
    Reads the driving_log.csv and stores in a pandas dataframe.
    Analyzes the steering info and returns a balanced dataset with
    a smaller subset of frames with ~ 0 steering angle

    :return: Pandas dataframe with balanced entries

    """
    A = pd.read_csv('data/driving_log.csv')
    turns = (A['steering'] >= 0.05) | (A['steering'] <= -0.05)
    A_turn = A[turns]         # turning
    A_straight = A[~turns]    # straight

    # Keep a few straight ones
    A_merge = pd.concat([A_turn,A_straight.sample(N_STRAIGHT)])

    # Shuffle rows
    A_merge = A_merge.sample(frac=1).reset_index(drop=True)

    A_train = A_merge[:-N_VAL]
    A_val = A_merge[-N_VAL:]

    if plot_hist:
        plt.subplot(2,1,1)
        plt.hist(A['steering'],bins=np.linspace(-1, 1, 50),color='b')
        plt.title('Raw dataset - histogram of steering angles')
        plt.subplot(2,1,2)
        plt.title('Balanced dataset - histogram of steering angles')
        plt.hist(A_merge['steering'],bins=np.linspace(-1,1,50),color='r')
        plt.show()

    return A_train, A_val

def val_data(A):
    N = len(A)
    x,y = [], []
    for i in A.index:
        path = os.path.join('data', A.center[i])
        xi = img_to_array(load_img(path))
        yi = A.steering[i]
        x.append(xi)
        y.append(A.steering[i])

    return np.array(x), np.array(y)

def data_generator(A,BATCH_SIZE):
    start = 0
    end = start + BATCH_SIZE
    while end < A.shape[0]:
        x, y = [], []
        for i in range(start, end):
            path = os.path.join('data', A.center[i])
            xi = img_to_array(load_img(path))
            yi = A.steering[i]
            x.append(xi)
            y.append(A.steering[i])

        start += BATCH_SIZE
        end += BATCH_SIZE
        if end > A.shape[0]:
            start = 0
            end = BATCH_SIZE
        yield np.array(x), np.array(y)


def nvidia():
    model = Sequential()

    # Layer 1
    model.add(BatchNormalization(input_shape=(160,320,3)))

    # Layer 2
    model.add(Convolution2D(24,5,5,border_mode='valid',subsample=(2,2)))

    # Layer 3
    model.add(Convolution2D(36,5,5,border_mode='valid',subsample=(2,2)))

    # Layer 4
    model.add(Convolution2D(48,5,5,border_mode='valid',subsample=(2,2)))

    # Layer 5
    model.add(Convolution2D(64,3,3,border_mode='valid',subsample=(1,1)))

    # Layer 6
    model.add(Convolution2D(64,3,3,border_mode='valid',subsample=(1,1)))

    # Layer 6a
    model.add(Flatten())
    #model.add(Dropout(0.5))

    # Layer 7
    model.add(Dense(100))

    # Layer 8
    model.add(Dense(50))

    # Layer 9
    model.add(Dense(10))

    # Output
    model.add(Dense(1, activation='linear'))

    # Training
    model.compile(loss='mse',optimizer='adam')
    return model



def train(FILE):
    net = nvidia()
    A_train,A_val = read_data()
    N = int(A_train.shape[0]/BATCH_SIZE)*BATCH_SIZE
    print("Number of examples available = {}".format(A_train.shape[0]))
    print("Batch size = {}".format(BATCH_SIZE))
    print("Samples per epoch = {}".format(N))
    T = data_generator(A_train,BATCH_SIZE)
    V = data_generator(A_val,BATCH_SIZE)
    net.fit_generator(T,samples_per_epoch=N,nb_epoch=NB_EPOCHS,validation_data=val_data(A_val),nb_val_samples=N_VAL)
    net.save(FILE)
    K.clear_session()

# def load(saved_file):
#     return load_model(saved_file)
#
#
# def evaluate(FILE):
#     net = load_model(FILE)
#     G = get_data(256)
#     x,y = next(G)
#     print(x.shape,y.shape)
#     output = net.predict_on_batch(x)
#     for y_act,y_pred in zip(y,output):
#         print("ACTUAL/PREDICTED  = {} {}".format(y_act,y_pred))
#     K.clear_session()


FILE='model.h5'
TURN_THRESHOLD = 0.1   # Threshold on steering angle to pick turns
N_STRAIGHT = 300        # Number of straight images to pick
N_VAL = 256
BATCH_SIZE = 256
NB_EPOCHS = 10

train(FILE)
evaluate(FILE)