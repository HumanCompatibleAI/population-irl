import numpy as np
from io import StringIO

from tempfile import TemporaryFile
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
#Set the Seed 
from gym.spaces.prng import seed
import gym

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])





# def seaquest(nsample = 100, oxigenlevel = 32, oxigenpenal = 5, swimmingman = 20, 
#           fishscore = 20,max_iter = 10**3):

  
nsample = 1
oxigenlevel = 32
oxigenpenal = 5
swimmingman = 20
fishscore = 20
max_iter = 10**3  

np.set_printoptions(threshold=np.nan)
env = gym.make('Seaquest-ram-v0')
env.reset()
env.seed(300)
env.render()
action = env.action_space.sample()
observation, reward, done, info = env.step(action)

oxi = 64
swi = 0
for i_episode in range(nsample):

    observation = env.reset()
    summ = 0
    for t in range(max_iter):

        action = env.action_space.sample()
        #print(action)
        observation, reward, done, info = env.step(action)

        if observation[102] < oxi and observation[102] < oxigenlevel:
        	summ -= oxigenpenal
       		print(summ)
        if observation[62] > swi:
        	summ += swimmingman
        	print(summ)
        swi = observation[62]
        oxi = observation[102]
        if reward > 0:
            summ += fishscore
            print(observation[62])
            print(observation[102])
            print(observation)
            gray = rgb2gray(observation) 
            #plt.imshow(gray, cmap = plt.get_cmap('gray'))
            #plt.savefig('fohhhh%d.png' % t)
            print("Hehe")
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print(summ)
            break
            env.close()
            gray = rgb2gray(K) 
            #plt.imshow(gray, cmap = plt.get_cmap('gray'))
            #plt.savefig('fo.png')

#np.savetxt('test1.txt', FACT, fmt='%d')
#@np.savetxt('test2.txt', actt, fmt='%d')
#np.savetxt('test3.txt', FOB, fmt='%d')