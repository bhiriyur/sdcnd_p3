from keras.models import Sequential, load_model
from keras.layers import Convolution2D, Dense, MaxPooling2D, Dropout, BatchNormalization, Flatten, Input, Lambda, ELU
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
    turns = (A['steering'] >= TURN_THRESHOLD) | (A['steering'] <= -TURN_THRESHOLD)
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

def get_image_data(A,i,mode):
    modes = {1:('center',0.0),
             2:('left', -0.3),
             3:('right', 0.3)}
    path = os.path.join('data',A[modes[mode][0]][i].strip())
    xi = img_to_array(load_img(path))
    yi = A.steering[i]+modes[mode][1]
    return xi,yi
    

def data_generator(A,BATCH_SIZE):
    start = 0
    end = start + BATCH_SIZE
    while end < A.shape[0]:
        x, y = [], []
        for i in range(start, end):
            #mode = np.random.randint(1,3)
            mode = 1
            xi,yi = get_image_data(A,i,mode)
            x.append(xi)
            y.append(yi)

           
        start += BATCH_SIZE
        end += BATCH_SIZE
        if end > A.shape[0]:
            start = 0
            end = BATCH_SIZE
        yield np.array(x), np.array(y)


def comma_ai(time_len=1):
    ch, row, col = 3, 160, 320  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    
    return model

def nvidia():
    model = Sequential()

    # Layer 1
    model.add(BatchNormalization(input_shape=(160,320,3)))

    # Layer 2
    model.add(Convolution2D(24,5,5,activation='elu',subsample=(2,2)))
    model.add(MaxPooling2D())

    # Layer 3
    model.add(Convolution2D(36,3,3,activation='elu',subsample=(2,2)))
    model.add(MaxPooling2D())

    # Layer 4
    model.add(Convolution2D(48,3,3,activation='elu',subsample=(2,2)))
    
    # Layer 5
    model.add(Convolution2D(64,3,3,activation='elu',subsample=(1,1)))


    # Layer 6a
    model.add(Flatten())
    model.add(Dropout(0.2))

    # Layer 7
    model.add(Dense(100,activation='elu'))
    model.add(Dropout(0.5))

    # Layer 8
    model.add(Dense(50,activation='elu'))
    model.add(Dropout(0.5))

    # Layer 9
    model.add(Dense(10,activation='elu'))

    # Output
    model.add(Dense(1, activation='linear'))

    # Training
    model.compile(loss='mse',optimizer='adam')
    return model



def train(FILE):
    net = nvidia()
    #net = comma_ai()
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


def evaluate(FILE):
    net = load_model(FILE)
    A_train,A_val = read_data()
    A_val = A_val.sample(frac=1).reset_index(drop=True)
    y_act, y_pred = [], []
    for i in range(len(A_val)):
        xi,yi = get_image_data(A_val,i,1)
        output = net.predict_on_batch(np.array([xi]))
        print("ACTUAL/PREDICTED  = {:8.4f} {:8.4f}".format(yi,output[0][0]))
        y_act.append(yi)
        y_pred.append(output[0][0])
    K.clear_session()
    
    plt.figure()
    plt.plot(y_act,name='Actual',color='b')
    plt.plot(y_act,name='Predicted',color='r')    
    plt.legend(loc='best')
    plt.show()


if __name__=='__main__':    
    
    FILE='model.h5'
    TURN_THRESHOLD = 0.05   # Threshold on steering angle to pick turns
    N_STRAIGHT = 200        # Number of straight images to pick
    N_VAL = 256
    BATCH_SIZE = 128
    NB_EPOCHS = 10
    train(FILE)
    evaluate(FILE)
