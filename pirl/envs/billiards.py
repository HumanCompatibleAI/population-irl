from gym import utils
from gym.utils import seeding
from gym.envs.mujoco import MujocoEnv
from matplotlib import cm
from mujoco_py import const
import numpy as np

from airl.envs.dynamic_mjc.model_builder import MJCModel

def billiards_model(num_cats, particle_size, cmap_name='Set1'):
    model = MJCModel('billiards')
    root = model.root

    # Setup
    root.compiler(angle='radian', inertiafromgeom='true', coordinate='local')
    root.option(timestep=0.01, gravity='0 0 0', iterations='20', integrator='Euler')
    default = root.default()
    default.joint(damping=1, limited='false')
    default.geom(contype=2, conaffinity=1, condim=1, friction='.5 .1 .1',
                 density='1000', margin='0.002')

    worldbody = model.root.worldbody()

    # Camera, looking from the top
    worldbody.camera(name='camera', mode='fixed', pos=[0.5, 0.5, 2])

    # Particle controlled by the agent
    particle = worldbody.body(name='particle', pos=[0, 0, 0])
    particle.geom(name='particle_geom', type='sphere', contype=1, conaffinity=1,
                  size=particle_size, rgba='1.0 1.0 1.0 1')
    particle.joint(name='ball_x', type='slide', pos=[0,0,0], axis=[1,0,0])
    particle.joint(name='ball_y', type='slide', pos=[0,0,0], axis=[0,1,0])

    # Targets the agent can hit
    cmap = cm.get_cmap(cmap_name)
    for i in range(num_cats):
        prefix = 'target{}'.format(i)
        target = worldbody.body(name=prefix, pos=[0, 0, 0])
        target.geom(name=prefix + '_geom', type='sphere', contype=1, conaffinity=1,
                    size=particle_size, rgba=list(cmap(i)))
        target.joint(name='ball{}_x'.format(i), type='slide', pos=[0, 0, 0],
                     axis=[1, 0, 0])
        target.joint(name='ball{}_y'.format(i), type='slide', pos=[0, 0, 0],
                     axis=[0, 1, 0])

    # Borders of the world
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    def pad(z):
        if z <= 0:
            return z - particle_size
        else:
            return z + particle_size
    for i in range(4):
        xs, ys = corners[i]
        xe, ye = corners[(i + 1) % 4]
        fromto = [pad(xs), pad(ys), particle_size / 2,
                  pad(xe), pad(ye), particle_size / 2]
        worldbody.geom(conaffinity=1, fromto=fromto, name='side{}'.format(i),
                       rgba='0.5 0.5 0.5 1', size=particle_size, type='capsule')

    actuator = model.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-1.0, 1.0], ctrllimited=True)
    actuator.motor(joint="ball_y", ctrlrange=[-1.0, 1.0], ctrllimited=True)

    return model


def create_reward(params, rng):
    params = np.array(params)
    means = params[:, 0]
    sds = params[:, 1]
    return means + rng.randn(params.shape[0]) * sds


def random_pos(n, gap, rng):
    '''Sample n points from [0,1]x[0,1] where all points are at least
       gap l_2 norm apart and gap/2 away from the end points.'''
    # Brute force solution. Could also sample from a grid, but less continuous.
    points = []
    for i in range(n):
        ok = False
        while not ok:
            p = rng.rand(2) * (1 - gap) + gap / 2
            ok = True
            for x in points:
                if np.linalg.norm(p - x) < gap:
                    ok = False
        points.append(p)
    return points


AGENT_ID = 4
class BilliardsEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, params, num_balls=None, particle_size=0.05, ctrl=0.1, seed=0):
        seed = seeding.create_seed(seed)
        rng = np.random.RandomState(seed)

        self.rewards = create_reward(params, rng)
        if num_balls is None:
            num_balls = len(self.rewards)
        self.num_targets = num_balls
        self.particle_size = particle_size
        self.ctrl = ctrl

        num_cats = len(self.rewards)
        assert num_cats >= num_balls
        model = billiards_model(num_cats, particle_size=particle_size)
        with model.asfile() as f:
            MujocoEnv.__init__(self, f.name, 5)
            utils.EzPickle.__init__(self, params, num_balls, particle_size, seed)

    def step(self, a):
        done = False
        reward = -self.ctrl * np.square(a).sum()  # control cost

        starting_state = (self.data.qpos == 0).all()  # before reset_model()
        self.do_simulation(a, self.frame_skip)

        num_contacts = self.sim.data.ncon
        if num_contacts > 0 and not starting_state:
            for contact in self.sim.data.contact[:num_contacts]:
                if contact.geom1 == AGENT_ID:
                    opp = contact.geom2 - (AGENT_ID + 1)
                elif contact.geom2 == AGENT_ID:
                    opp = contact.geom1 - (AGENT_ID + 1)
                else:  # collision didn't involve agent
                    continue
                if 0 <= opp < len(self.rewards):
                    # SOMEDAY: for collisions involving multiple balls,
                    # picking the first one in the list may lead to chaotic
                    # behavior.
                    done = True
                    reward += self.rewards[opp]

        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        our_pos = self.sim.data.qpos.flat[0:2]
        our_vel = self.sim.data.qvel.flat[0:2]
        target_pos = self.sim.data.qpos.flat[2:].reshape(-1, 2)
        live_targets = np.array(target_pos >= 0, dtype=bool).all(1, keepdims=True)
        target_pos = np.where(live_targets, target_pos, 0.5)
        return np.concatenate([
            our_pos,
            our_vel,
            target_pos.flat,
            (target_pos - our_pos).flat, # Is this the best way to do this?
            live_targets.flat,
        ])

    def reset_model(self):
        num_cats = len(self.rewards)
        active_cats = self.np_random.choice(num_cats, self.num_targets, replace=False)

        # particles must be at least diameter = self.particle_size * 2 apart
        # to avoid contact at initialization; add some extra margin as MuJoCo
        # has some numerical error in contact detection.
        gap = self.particle_size * 2.1
        active_pos = random_pos(self.num_targets + 1, gap, self.np_random)
        active_pos = np.array(active_pos)

        qpos = np.zeros((num_cats + 1, 2))
        # inactive target balls, space them out to avoid contact forces
        qpos[:, :] = (-np.arange(num_cats + 1) - 1).reshape(-1, 1)
        qpos[0, :] = active_pos[0]  # agent ball
        for i in range(self.num_targets):  # active target balls
            cat = active_cats[i]
            qpos[cat + 1] = active_pos[i + 1]
        qpos = np.array(qpos).flatten()
        qvel = np.zeros(2 * (num_cats + 1))
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.fixedcamid = 0
        self.viewer.cam.type = const.CAMERA_FIXED
