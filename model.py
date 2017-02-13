from keras.models import Sequential, load_model
from keras.layers import Convolution2D, Dense, MaxPooling2D, Dropout, BatchNormalization
from keras.layers import Flatten, Input, Lambda, ELU, Cropping2D
from keras.preprocessing.image import img_to_array, load_img, flip_axis, random_shift
from keras.optimizers import Adam
from keras import backend as K
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

def read_data(plot_hist=True):
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
    A_merge = pd.concat([A_turn,A_straight.sample(frac=(1-DROP_THRESHOLD))])

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

def get_image_data(A,i,mode,flip=0,vshift=0.0,hshift=0.0):
    modes = {1:('center',0.0),
             2:('left',  0.3),
             3:('right',-0.3)}
    path = os.path.join('data',A[modes[mode][0]][i].strip())
    xi = img_to_array(load_img(path))
    xi = random_shift(xi,vshift,hshift,0,1,2)
    yi = A.steering[i]+modes[mode][1]
    if flip==1:
        xi = flip_axis(xi,1)
        yi = -yi

    return xi,yi
    

def data_generator(A,BATCH_SIZE):
    while True:
        x, y = [], []
        count = 0
        while count < BATCH_SIZE:
            # Select an image by random
            i = np.random.randint(0,len(A))

            # Drop the selection with a certain probability if it's straight
            #if (abs(A['steering'][i]) < TURN_THRESHOLD) and (np.random.uniform()<=DROP_THRESHOLD):
            #    continue
            
            # Pick center (prob = 67%), left (16%) or right (16%) image 
            mode = np.random.choice([1,1,1,1,2,3],1)

            # If center camera image, flip with 50% probability
            flip = np.random.randint(0,2)
            if mode[0] != 1: flip = 0

            # Random shift in width and height
            wshift,hshift = 0.2*np.random.random(2)
            xi,yi = get_image_data(A,i,mode[0],flip,wshift,0.0*hshift)
            x.append(xi)
            y.append(yi)

            count += 1
           
        yield np.array(x), np.array(y)

def hist_A(A,BATCH_SIZE,N):
    """
    Generate an epoch's worth of training data from
    the generator and plot the histogram of steering angles
    """

    T = data_generator(A,BATCH_SIZE)
    y = []
    for i in tqdm(range(int(N/BATCH_SIZE))):
        _,yi = next(T)
        y.append(yi)

    plt.hist(np.concatenate(y),bins=np.linspace(-1, 1, 50),color='b')
    plt.show()
    

def comma_ai(time_len=1):
    """
    A variant of the comma.ai model
    """
    
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
    """
    A variant of the nvidia model
    """
    model = Sequential()

    # Crop layer
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    print("LAYER: {:30s} {}".format('Cropping',model.layers[-1].output_shape))

    # Layer 1
    model.add(BatchNormalization())
    print("LAYER: {:30s} {}".format('Batch-Normalization',model.layers[-1].output_shape))         

    # Layer 2
    model.add(Convolution2D(24,3,3,border_mode='same',activation='elu',subsample=(2,2)))
    print("LAYER: {:30s} {}".format('Conv2D-24-3x3-s2',model.layers[-1].output_shape))    
    model.add(MaxPooling2D())
    print("LAYER: {:30s} {}".format('Maxpool2D',model.layers[-1].output_shape))    

    # Layer 3
    model.add(Convolution2D(36,3,3,border_mode='same',activation='elu',subsample=(2,2)))
    print("LAYER: {:30s} {}".format('Conv2D-36-3x3-s2',model.layers[-1].output_shape))        
    model.add(MaxPooling2D())
    print("LAYER: {:30s} {}".format('Maxpool2D',model.layers[-1].output_shape))
    
    # Layer 4
    model.add(Convolution2D(48,3,3,border_mode='same',activation='elu',subsample=(1,1)))
    print("LAYER: {:30s} {}".format('Conv2D-48-3x3-s1',model.layers[-1].output_shape))        
    print("LAYER: {:30s} {}".format('Maxpool2D',model.layers[-1].output_shape))
    
    # Layer 5
    model.add(Convolution2D(64,3,3,activation='elu',subsample=(1,1)))
    print("LAYER: {:30s} {}".format('Conv2D-64-3x3-s1',model.layers[-1].output_shape))        

    # Layer 6a
    model.add(Flatten())
    model.add(Dropout(0.5))
    print("LAYER: {:30s} {}".format('Flatten',model.layers[-1].output_shape))    

    
    # Layer 7
    model.add(Dense(500,activation='elu'))
    model.add(Dropout(0.5))
    print("LAYER: {:30s} {}".format('FC',model.layers[-1].output_shape))        


    # Layer 8
    model.add(Dense(100,activation='elu'))
    model.add(Dropout(0.5))
    print("LAYER: {:30s} {}".format('FC',model.layers[-1].output_shape))        

    # Layer 9
    model.add(Dense(10,activation='elu'))
    print("LAYER: {:30s} {}".format('FC',model.layers[-1].output_shape))        

    # Output
    model.add(Dense(1, activation='linear'))
    print("LAYER: {:30s} {}".format('OUTPUT',model.layers[-1].output_shape))        

    # Training
    adamopt = Adam(lr=0.0001)
    model.compile(loss='mse',optimizer=adamopt)
    return model


def train(FILE):
    """
    Build and train the network
    """
    net = nvidia()
    #net = comma_ai()

    A_train,A_val = read_data()

    print("Number of examples available = {}".format(A_train.shape[0]))
    print("Batch size = {}".format(BATCH_SIZE))
    print("Samples per epoch = {}".format(N_SAMPLE))
    hist_A(A_train,BATCH_SIZE,N_SAMPLE)
    
    T = data_generator(A_train,BATCH_SIZE)
    net.fit_generator(T, samples_per_epoch=N_SAMPLE, nb_epoch=NB_EPOCHS,
                      validation_data=val_data(A_val), nb_val_samples=N_VAL)
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
    plt.plot(y_act,label='Actual',color='b')
    plt.plot(y_pred,label='Predicted',color='r')
    plt.legend(loc='best')
    plt.show()


if __name__=='__main__':    
    
    FILE='model.h5'
    TURN_THRESHOLD = 0.05   # Threshold on steering angle to pick turns
    DROP_THRESHOLD = 0.90   # Straight/Turning drop threshold
    N_VAL = 256
    BATCH_SIZE = 128
    N_SAMPLE = BATCH_SIZE*100
    NB_EPOCHS = 4
    train(FILE)
    evaluate(FILE)
