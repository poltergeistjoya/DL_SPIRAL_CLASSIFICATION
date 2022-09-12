import matplotlib.pyplot as plt
import numpy as np

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

#feed in x, y, and category as training data, predict boundaries

#PLOTTING
fig, ax = plt.subplots(1,1, figsize=(15,15), dpi = 200)
ax.set_title("Spirals")
ax.set_xlabel("Spiral Radius")
ax.set_ylabel("Spiral Radius")
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)

ax.plot(x1,y1, "o", x2,y2, "o")

plt.savefig("./spirals.pdf")


