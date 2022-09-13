import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#random number generator
rng = np.random.default_rng()
#Generate spiral data, vary r while moving through x and y
clean_r = np.linspace(1,12, 1000)
clean_theta1=np.linspace(6,2.5,1000)
clean_theta2 = np.linspace(5, 1.5, 1000)

#make noisy draw fromm normal dist
r = rng.normal(loc= clean_r, scale =0.1)
theta1 = rng.normal(loc= clean_theta1, scale =0.1)
theta2 = rng.normal(loc= clean_theta2, scale =0.1)

x1=r*np.cos(np.pi*theta1)
y1=r*np.sin(np.pi*theta1)

x2=r*np.cos(np.pi*theta2)
y2=r*np.sin(np.pi*theta2)

#make relu activation function
#not sure how the sizes will work here
def relu(x,y):
    if argmax(0,x) > 0:
        return y
    else
        return 0

#feed in x, y, and category as training data, predict boundaries with multilayer perceptron
class Model(tf.Module):
        #variables to be tuned in inits
        #initialize 3 hidden layers with appropriate sizes for dimensionality
        #SHOULD THE OUTPUT LAYER BE IN HERE?? YES RIGHT CUZ IT ALSO HAS TUNABLE WEIGHTS
    def __init__(self):
        self.layer1 = tf.Variable(rng(normal(shape[2,6])))
        self.layer2 = tf.Variable(rng(normal(shape[6,6])))
        self.layer3 = tf.Variable(rng(normal(shape[6,6])))
        self.output = tf.Variable(rng(normal(shape[6,2])))

        #this makes output, multiply by sigmoid in the end
    def __call__(self):
        #pass input through relu
        #DO I MAKE A NEW SET OF ALL THE X1Y1 X2 Y2 FROM BEFORE ? OR DO I JUST PASS THEM ALL THROUGH HOW DO I DO THAT??/
        relu(x,y)
        #last layer activated by sigmoid ? passed through sigmoid



#PLOTTING
fig, ax = plt.subplots(1,1, figsize=(15,15), dpi = 200)
ax.set_title("Spirals")
ax.set_xlabel("Spiral Radius")
ax.set_ylabel("Spiral Radius")
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)

ax.plot(x1,y1, "o", x2,y2, "o")

plt.savefig("./spirals.pdf")

#plot with contour map instead not scatterplot


#papers say don't use L2 with ADAM, use weight decay instead
