'''Code to implement a toy model of Wang Lamndau sampling'''

import numpy as np
import random as rnd
import matplotlib.pyplot as plt

f = 2.5  # the initial multiplicative factor
epsilon = 1.0001
g_e = np.array([0.05, 0.70, 0.15, 0.10])
histogram = np.zeros(4)
dens_estimate = np.ones(4)


def prop_config(g):
    i = rnd.random()
    if i < g[0]:
        new_conf = 0
    elif i < g[0] + g[1]:
        new_conf = 1
    elif i < g[0] + g[1] + g[2]:
        new_conf = 2
    else:
        new_conf = 3
    return new_conf


def is_flat(H):
    avg = np.average(H)
    std = np.std(H)
    if std / avg < 0.03:
        return True
    else:
        return False


def accept_config(H, g, next_step):
    H[next_step] += 1
    g[next_step] *= f


current_config = prop_config(g_e)

index = 1
convergence=[]
while (f > epsilon):
    next_config = prop_config(g_e)
    if rnd.random() < dens_estimate[current_config] / dens_estimate[next_config]:
        current_config = next_config
        accept_config(histogram, dens_estimate, current_config)
    else:
        accept_config(histogram, dens_estimate, current_config)
    index += 1
    if is_flat(histogram):
        print(dens_estimate/np.sum(dens_estimate))
        print(histogram)
        convergence.append(dens_estimate[2]/np.sum(dens_estimate))
        histogram = np.zeros(len(g_e))
        f = np.sqrt(f)
        index = 1

x=np.linspace(0,13,100)
y=np.array([0.15 for i in range(100)])
plt.plot(x,y)
plt.plot(convergence)
plt.show()


