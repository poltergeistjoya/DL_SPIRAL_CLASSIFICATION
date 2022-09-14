#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

from absl import flags


#All flags
#WHAT IS DEBUG USED FOR????
FLAGS = flags.FLAGS
flags.DEFINE_integer("num_samples", 200, "Number of samples in dataset")
flags.DEFINE_integer("batch_size", 50, "Number of samples in batch")
flags.DEFINE_integer("num_iters", 300, "Number of forward/backward pass iterations")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate/initial step size")
flags.DEFINE_integer("random_seed", 31415, "Random seed for reproducible results")
flags.DEFINE_float("sigma_noise", 0.5, "Standard deviation of noise random variable")

#make layer class
class Layer(tf.Module):
    print("make layer class")


#feed in x, y, and category as training data, predict boundaries with multilayer perceptron
class Model(tf.Module):
        #variables to be tuned in inits
        #initialize 3 hidden layers with appropriate sizes for dimensionality
        #SHOULD THE OUTPUT LAYER BE IN HERE?? YES RIGHT CUZ IT ALSO HAS TUNABLE WEIGHTS
    def __init__(self):
        print("make model with layer class")

        #this makes output, multiply by sigmoid in the end
    def __call__(self):
        #pass input through relu
        #DO I MAKE A NEW SET OF ALL THE X1Y1 X2 Y2 FROM BEFORE ? OR DO I JUST PASS THEM ALL THROUGH HOW DO I DO THAT??/
        #last layer activated by sigmoid ? passed through sigmoid
        print("hello")

def main():

    #parse flags before we use them
    FLAGS(sys.argv)

    #set seed for reproducible results
    seed_sequence = np.random.SeedSequence(FLAGS.random_seed)
    np_seed, tf_seed = seed_sequence.spawn(2) #spawn 2 sequences for 2 threads

    #why do we need two different random seed sequences?
    np_rng =np.random.default_rng(np_seed)
    tf_rng = tf.random.Generator.from_seed(tf_seed.entropy)

    #Generate spiral data, vary r while moving through x and y
    clean_r = np.linspace(1,12, FLAGS.num_samples)
    clean_theta1=np.linspace(6,2.5, FLAGS.num_samples)
    clean_theta2 = np.linspace(5, 1.5, FLAGS.num_samples)

    #make noisy draw from normal dist
    r = np_rng.normal(loc= clean_r, scale =0.1)
    theta1 = np_rng.normal(loc= clean_theta1, scale =0.1)
    theta2 = np_rng.normal(loc= clean_theta2, scale =0.1)

    x1=r*np.cos(np.pi*theta1)
    y1=r*np.sin(np.pi*theta1)

    x2=r*np.cos(np.pi*theta2)
    y2=r*np.sin(np.pi*theta2)

    #Combine spiral data to make one training set
    data_x = np.append(x1,x2)
    data_y = np.append(y1,y2)

    #PLOTTING
    fig, ax = plt.subplots(1,1, figsize=(15,15), dpi = 200)
    ax.set_title("Spirals")
    ax.set_xlabel("Spiral Radius")
    ax.set_ylabel("Spiral Radius")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.plot(x1, y1, "o", x2, y2, "o")

    plt.savefig("./spiralstest.pdf")

    #plot with contour map instead not scatterplot

#this makes sure the main function runs first
if __name__ == "__main__":
    main()
