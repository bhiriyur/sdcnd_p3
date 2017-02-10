import pandas as pd
import matplotlib.pyplot as plt
import os
from random import randint

A = pd.read_csv('driving_log.csv')
print(os.getcwd())
for k in range(20):
    i = randint(0,A.shape[0])
    cfile = A['center'][i].strip()
    lfile = A['left'][i].strip()
    rfile = A['right'][i].strip()
    steer = A['steering'][i]
    throttle = A['throttle'][i]
    brake= A['brake'][i]
    speed= A['speed'][i]
    #print("File {:3}, | Steering = {} Throttle = {} Brake = {} Speed = {}".format(i,steer,throttle,brake,speed))
    plt.figure(figsize=(5,9))
    plt.subplot(3,1,1)
    plt.imshow(plt.imread(cfile))
    plt.title("{:3}, | steer:{:4.2f} speed:{:4.2f}".format(i,steer,throttle,brake,speed))
    plt.axis('off')
    plt.subplot(3,1,2)
    plt.imshow(plt.imread(lfile))
    plt.axis('off')
    plt.subplot(3,1,3)
    plt.imshow(plt.imread(rfile))
    plt.axis('off')
    plt.show()