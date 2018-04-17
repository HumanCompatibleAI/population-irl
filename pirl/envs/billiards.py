from gym import utils
from gym.utils import seeding
from gym.envs.mujoco import MujocoEnv
import numpy as np

from airl.envs.dynamic_mjc.model_builder import MJCModel

def billiards_model(num_balls, particle_size=0.01):
    model = MJCModel('billiards')
    root = model.root

    # Setup
    root.compiler(angle='radian', inertiafromgeom='true', coordinate='local')
    root.option(timestep=0.01, gravity='0 0 0', iterations='20', integrator='Euler')
    default = root.default()
    default.joint(damping=1, limited='false')
    default.geom(contype=2, conaffinity='1', condim='1', friction='.5 .1 .1',
                 density='1000', margin='0.002')

    worldbody = model.root.worldbody()

    # Particle controlled by the agent
    particle = worldbody.body(name='particle', pos=[0.5, 0, 0])
    particle.geom(name='particle_geom', type='sphere',
                  size=particle_size, rgba='0.0 0.0 1.0 1', contype=1)
    particle.site(name='particle_site', pos=[0,0,0], size=0.01)
    particle.joint(name='ball_x', type='slide', pos=[0,0,0], axis=[1,0,0])
    particle.joint(name='ball_y', type='slide', pos=[0,0,0], axis=[0,1,0])

    # Targets the agent can hit
    for i in range(num_balls):
        prefix = 'target{}'.format(i)
        target = worldbody.body(name=prefix, pos=[i*0.02, 0, 0])
        target.geom(name=prefix + '_geom', conaffinity=2, type='sphere',
                    size=particle_size, rgba=[0,0.9,0.1,1])

    # Borders of the world
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    def pad(z):
        if z <= 0:
            return z - particle_size / 2
        else:
            return z + particle_size / 2
    for i in range(4):
        xs, ys = corners[i]
        xe, ye = corners[(i + 1) % 4]
        fromto = [pad(xs), pad(ys), particle_size,
                  pad(xe), pad(ye), particle_size]
        worldbody.geom(conaffinity=1, fromto=fromto, name='side{}'.format(i),
                       rgba='0.9 0.4 0.6 1', size=particle_size, type='capsule')

    actuator = model.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-1.0, 1.0], ctrllimited=True)
    actuator.motor(joint="ball_y", ctrlrange=[-1.0, 1.0], ctrllimited=True)

    return model


class BilliardsEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, num_balls, seed=0):
        self.init_seed = seeding.create_seed(seed)
        self.num_balls = num_balls
        model = billiards_model(num_balls)
        with model.asfile() as f:
            MujocoEnv.__init__(self, f.name, 5)

    def step(self, a):
        # TODO: implement
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        reward = 0
        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        # TODO: implement
        return np.array([])

    def reset_model(self):
        dim = 2 * (self.num_balls + 1)
        qpos = self.np_random.rand(dim)
        qvel = np.zeros(dim)
        self.set_state(qpos, qvel)

    def viewer_setup(self):
        # TODO: optional, but may be useful
        pass