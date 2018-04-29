import random
from random import randrange, uniform
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from rllab.misc import logger


from airl.envs.dynamic_mjc.model_builder import MJCModel


LEFT = 0
RIGHT = 1

def point_mass_maze(direction=RIGHT, length=1.2, borders=True, nlava = 4, nWater = 1, nTrees =1):
    '''
    We are creating a the Two-d maze conatins objects like Trees and Waters. They are both cylinder. We also conatins
    Lavas which looks like boxes.
    '''
    mjcmodel = MJCModel('twod_maze')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler")
    default = mjcmodel.root.default()
    default.joint(damping=1, limited='false')
    default.geom(friction=".5 .1 .1", density="1000", margin="0.002", condim="1", contype="2", conaffinity="1")
    worldbody = mjcmodel.root.worldbody()
    

    #We create index and size to keep track of sizes
    lavaindex = {}
    lavasize = {}
    Waterindex = {}
    Watersize = {}
    Treesindex = {}
    Treessize = {}

    # We have a bounch of lava in a fixed region and it can be customized by its numbers(nlava)
    #Should length be square
    for i in range(nlava):
        # We may have to set seed here.
        lavaindex[i] = [random.random()*length,random.random()*length,0]
        lavasize[i] = [random.random()*length/(5*nlava), random.random()*length/(5*nlava), 0.03]
        exec("lava{} = worldbody.body(name='lava' + str({}), pos=lavaindex[{}])".format(i, i, i)) 
        exec("lava{}.geom(name='lava_geom'+ str({}), conaffinity=2, type='box', size=lavasize[{}], rgba=[0.2,0.2,0.8,1]) ".format(i, i, i)) 

    

    points = np.array([(random.random()*(length -0.04) + 0.02, random.random()*(length-0.04) + 0.02, 0) for i in range(1000)])
    #To make sure all those potential target and particle are outside of the lava region
    for i in range(nlava):
        ur = np.add(np.array(lavaindex[i]), np.array(lavasize[i])/2, np.array([length/10, length/10, 0])) #Upper Right
        ll = np.array(lavaindex[i]) - np.array(lavasize[i])/2 - np.array([length/10, length/10, 0])
        fal3 = np.all(np.logical_and(ll <= points, points <= ur), axis=1)

    points = points[np.logical_not(fal3)]


    # We have a bounch of waters in a fixed region and it can be customized by its numbers(nWater)
    for i in range(nWater):
        # We may have to set seed here.
        Waterindex[i] = [random.random()*length,random.random()*length,0]
        Watersize[i] = [random.random()*length/(3*nWater), random.random()*length/(10*nWater),0 ]
        exec("Water{} = worldbody.body(name='Water' + str({}), pos=Waterindex[{}])".format(i, i, i)) 
        exec("Water{}.geom(name='Water_geom'+ str({}), conaffinity=2, type='cylinder', size=Watersize[{}], rgba=[0.1,0.9,0.9,1]) ".format(i, i, i)) 
    
    


    for i in range(nWater):
        ur = np.add(np.array(lavaindex[i]), np.array(lavasize[i])/2, np.array([length/10, length/10, 0])) #Upper Right
        ll = np.array(lavaindex[i]) - np.array(lavasize[i])/2 - np.array([length/10, length/10, 0])
        fal2 = np.all(np.logical_and(ll <= points, points <= ur), axis=1)
    points = points[np.logical_not(fal2)]


    # We have a bounch of trees in a fixed region and it can be customized by its numbers(nTrees)
    for i in range(nTrees):
        # We may have to set seed here.
        Treesindex[i] = [random.random()*length,random.random()*length,0]
        Treessize[i] = [random.random()*length/(10*nTrees), random.random()*length/(3*nTrees), 0]
        exec("Trees{} = worldbody.body(name='Trees' + str({}), pos=Treesindex[{}])".format(i, i, i)) 
        exec("Trees{}.geom(name='Trees_geom'+ str({}), conaffinity=2, type='cylinder', size=Treessize[{}], rgba=[0.1,0.9,0.3,1]) ".format(i, i, i)) 

    # We should ranomize the 
    
    for i in range(nTrees):
        ur = np.add(np.array(Treesindex[i]), np.array(Treessize[i])/2, np.array([length/10, length/10, 0])) #Upper Right
        ll = np.array(Treesindex[i]) - np.array(Treessize[i])/2 - np.array([length/10, length/10, 0])
        fal1 = np.all(np.logical_and(ll <= points, points <= ur), axis=1)


    outbox = points[np.logical_not(fal1)].tolist()
    particle = worldbody.body(name='particle', pos=outbox[0])
    particle.geom(name='particle_geom', type='sphere', size='0.03', rgba='0.0 0.0 1.0 1', contype=1)
    particle.site(name='particle_site', pos=[0,0,0], size=0.01)
    particle.joint(name='ball_x', type='slide', pos=[0,0,0], axis=[1,0,0])
    particle.joint(name='ball_y', type='slide', pos=[0,0,0], axis=[0,1,0])

    #We should randomize it within some region
    for i in range(1, len(outbox)):
        if np.linalg.norm(np.asarray(outbox[i])-np.asarray(outbox[0]) )> 0.03:
            break
    target = worldbody.body(name='target', pos=outbox[1])
    target.geom(name='target_geom', conaffinity=2, type='sphere', size=0.02, rgba=[0.9, 0.1,0.1,1])

    L = -0.1
    R = length
    U = length
    D = -0.1

    if borders:
        worldbody.geom(conaffinity=1, fromto=[L, D, .01, R, D, .01], name="sideS", rgba="0.9 0.4 0.6 1", size=".02", type="capsule")
        worldbody.geom(conaffinity=1, fromto=[R, D, .01, R, U, .01], name="sideE", rgba="0.9 0.4 0.6 1", size=".02", type="capsule")
        worldbody.geom(conaffinity=1, fromto=[L, U, .01, R, U, .01], name="sideN", rgba="0.9 0.4 0.6 1", size=".02", type="capsule")
        worldbody.geom(conaffinity=1, fromto=[L, D, .01, L, U, .01], name="sideW", rgba="0.9 0.4 0.6 1", size=".02", type="capsule")

    # arena
    if direction == LEFT:
        BL = -0.1
        BR = length * 2/3
        BH = length/2
    else:
        BL = length * 1/3
        BR = length
        BH = length/2

    worldbody.geom(conaffinity=1, fromto=[BL, BH, .01, BR, BH, .01], name="barrier", rgba="0.9 0.4 0.6 1", size=".02", type="capsule")

    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-1.0, 1.0], ctrllimited=True)
    actuator.motor(joint="ball_y", ctrlrange=[-1.0, 1.0], ctrllimited=True)

    return mjcmodel




class PointMazeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, direction=1, maze_length=0.6, sparse_reward=False, no_reward=False, episode_length=100, nlava = 4, nWater = 1, nTrees = 1):
        utils.EzPickle.__init__(self)
        self.sparse_reward = sparse_reward
        self.no_reward = no_reward
        self.max_episode_length = episode_length
        self.direction = direction
        self.length = maze_length
        self.nlava = nlava
        self.nWater = nWater
        self.nTrees = nTrees
        self.episode_length = 0
        #The dictionary we use to keep track of rewards
        self.reward = {}

        model = point_mass_maze(direction=self.direction, length=self.length)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)

    
    def step(self, a):
        vec_dist = self.get_body_com("particle") - self.get_body_com("target")

        reward_dist = - np.linalg.norm(vec_dist)  # particle to target
        reward_ctrl = - np.square(a).sum()
        #Penalty for close to or inside the lava:
        reward_lava = 0
        #Reward for close to or inside the water:
        reward_water = 0
        #Reward for close to or inside the Trees:
        reward_tree = 0
        for i in range(self.nlava) :
            #print(i)

            if "lava" + str(i) not in self.reward:
                self.reward["lava" + str(i)] = uniform(0, 10)
                break 
            

            dist = np.linalg.norm(self.get_body_com("particle") - self.get_body_com("lava" + str(i))) # Not sure how to call a list here

            #We give a penalty whenever it is closed to lava
            if dist < 0.01:


                reward_lava += (0.01 - dist)
        
        for i in range(self.nWater) :
            #print(i)

            if "Water" + str(i) not in self.reward:
                self.reward["Water" + str(i)] = uniform(0, 5)
                break 
            

            dist = np.linalg.norm(self.get_body_com("particle") - self.get_body_com("Water" + str(i))) # Not sure how to call a list here

            if dist < 0.01:
                # We just give it rewards the first time it hits water
                reward_water += self.reward["Water" + str(i)]
                self.reward["Water" + str(i)] = 0


                

        for i in range(self.nTrees) :
            #print(i)

            if "Trees" + str(i) not in self.reward:
                self.reward["Trees" + str(i)] = uniform(-1, 2)
                break 
            

            dist = np.linalg.norm(self.get_body_com("particle") - self.get_body_com("Trees" + str(i))) # Not sure how to call a list here

            if dist < 0.01:
                reward_tree += self.reward["Trees" + str(i)]
                # We just give it rewards the first time it hits trees
                self.reward["Trees" + str(i)] = 0


        reward_live = self.episode_length * 0.01 #
        if self.no_reward:
            reward = 0
        elif self.sparse_reward:
            if reward_dist <= 0.1:
                reward = 1
            else:
                reward = 0
        else:
            reward = reward_dist + 0.01 * reward_ctrl - reward_lava  - reward_live - reward_tree - reward_water

        self.do_simulation(10*a, self.frame_skip)
        ob = self._get_obs()
        self.episode_length += 1
        done = np.linalg.norm(vec_dist) < 0.3

        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    
    def reset_model(self):
        qpos = self.init_qpos
        self.episode_length = 0
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        self.episode_length = 0
        self.reward = {}
        return self._get_obs()

    
    def _get_obs(self):
        return np.concatenate([
            self.get_body_com("particle"),
            #self.get_body_com("target"),
        ])

    
    def plot_trajs(self, *args, **kwargs):
        pass

    
    def log_diagnostics(self, paths):
        rew_dist = np.array([traj['env_infos']['reward_dist'] for traj in paths])
        rew_ctrl = np.array([traj['env_infos']['reward_ctrl'] for traj in paths])

        logger.record_tabular('AvgObjectToGoalDist', -np.mean(rew_dist.mean()))
        logger.record_tabular('AvgControlCost', -np.mean(rew_ctrl.mean()))
        logger.record_tabular('AvgMinToGoalDist', np.mean(np.min(-rew_dist, axis=1)))
