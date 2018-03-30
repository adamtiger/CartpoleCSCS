import gym
import numpy as np
from skimage import transform as trf
import random
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten
from keras.optimizers import Adam, SGD, RMSprop
from dataprovider import Optimizer as opt
from log import Logger, Mode


class DqnBase:

    def __init__(self, params):
        self.process_id = params.id
        self.env = None
        self.buffer_size = params.buf_size
        self.batch_size = params.batch
        self.epoch = params.epoch
        self.max_episode = params.max_ep
        self.eps0 = params.eps
        self.gamma = params.gamma
        self.C = params.C
        self.train_freq = params.train_freq
        self.eval_freq = params.eval_freq
        self.weight_file_name = params.w_file_name
        self.net = params.net
        self.lr = params.lr
        self.opt = params.opt

        self.q_cont, self.q_frzn = None, None

        self.buffer = []
        self.log = Logger(params)

    def train_function(self):

        self._init_buffer(self.batch_size)

        self.log.log(Mode.STD_LOG, "Initialization was finished.")
        self.log.log(Mode.STD_LOG, "Training was started.")

        ep_id = 1
        cntr = 0
        eval_permitted = True
        rtn = 0
        exps = []

        eps = self.eps0
        self.env.reset()
        obs, _, done, _ = self.env.step(0)

        while ep_id < self.max_episode:

            cntr += 1

            if done:
                if ep_id % 10 == 0:
                    self.log.log(Mode.STDOUT, 'Proc.Id: ' + str(self.process_id) + ' ' + str(ep_id))
                    self.log.log(Mode.LOG_F, 'Episode: ' + str(ep_id) + ' return: ' + str(rtn))
                rtn = 0
                ep_id += 1
                eval_permitted = True
                obs = self.env.reset()

            action = self.select_action_epsilon(obs, eps)
            obs_next, rw, done, _ = self.env.step(action)
            rtn += rw

            if done:
                if rtn < 180:
                    rw = -1
                    obs_next *= 0.0
                    obs *= 0.0
                elif rtn >= 180:
                    rw = 100

            exps.append((obs, rw, action, done, obs_next))
            obs = obs_next

            if cntr % 128 == 0:
                self.append(exps)
                exps.clear()

            if  cntr % self.train_freq == 0:
                x, y = self.sample(self.batch_size)
                self.q_cont.fit(x, y, epochs=self.epoch, batch_size=self.batch_size, verbose=0)

            if cntr % self.C == 0:
                self.q_frzn.set_weights(self.q_cont.get_weights())

            if ep_id % self.eval_freq == 0 and eval_permitted:
                r = self.evaluation()
                self.log.log(Mode.RET_F, [cntr, ep_id, r])
                eval_permitted = False

            eps = max(eps - 0.0001, 0.1)

        self.q_cont.save_weights(self.weight_file_name)
        self.log.log(Mode.STD_LOG, "Training was finished.")

    def evaluation(self):

        def avg(alist):
            n = len(alist)
            szum = 0
            for x in alist:
                szum += x

            return float(szum) / n

        self.log.log(Mode.STDOUT, 'Proc.Id: ' + str(self.process_id) + ' - evaluation started.')

        obs = self.env.reset()
        done = False
        rtn = 0
        ep_id = 0
        rtns = []

        while ep_id < 50:

            if done:
                rtns.append(rtn)
                rtn = 0
                ep_id += 1
                obs = self.env.reset()

            action = self.select_action(obs)
            obs, rw, done, _ = self.env.step(action)
            rtn += rw

        return avg(rtns)

    # ------------------------------------------------------
    # Functions for handling the buffer (experience replay)

    def clear_buffer(self):
        self.buffer.clear()

    def append(self, experiences):
        return

    def sample(self, number):
        exps = random.sample(self.buffer, number)  # experiences list
        obs = np.stack([x[0] for x in exps], axis=0)
        rws = np.array([x[1] for x in exps])
        acts = np.array([x[2] for x in exps])
        dones = np.array([x[3] for x in exps])
        next_obs = np.stack([x[4] for x in exps], axis=0)

        q_vals = self.q_cont.predict(obs, batch_size=number)  # (batch_size, 2)
        fzn_q_vals = self.q_frzn.predict(next_obs, batch_size=number)

        sub_values = rws + self.gamma * (1 - dones) * np.max(fzn_q_vals, axis=1)

        q_vals[list(range(number)), acts] = sub_values  # this will be the target in the network

        x = obs.astype(dtype=np.float32)
        y = q_vals.astype(dtype=np.float32)

        return x, y

    # ------------------------------------------------------
    # Functions for action selections

    # with epsilon-greedy
    def select_action_epsilon(self, state, eps):  # state shape: (4) nunmpy array
        s = np.expand_dims(state, axis=0)
        max_idx = np.argmax(self.q_cont.predict(s, batch_size=1))
        if random.random() < 1 - eps:
            return max_idx
        return (max_idx + 1) % 2

    # without epsilon-greedy
    def select_action(self, state):
        s = np.expand_dims(state, axis=0)
        return np.argmax(self.q_cont.predict(s, batch_size=1))

    # ------------------------------------------------------
    # Helper functions for initialization tasks

    def _init_optimizer(self):

        optz = None
        if self.opt == opt.ADAM:
            optz = Adam(self.lr)
        elif self.opt == opt.SGD:
            optz = SGD(self.lr)
        elif self.opt == opt.RMSPROP:
            optz = RMSprop(self.lr)

        return optz

    def _init_models(self):

        return

    def _init_buffer(self, number):

        exps = []
        obs, rw, done, _ = self.env.step(0)
        for _ in range(number):

            if done:
                obs = self.env.reset()

            action = self.env.action_space.sample()
            obs_next, rw, done, _ = self.env.step(action)
            exps.append((obs, rw, action, done, obs_next))
            obs = obs_next

        self.append(exps)


class DqnLow(DqnBase):

    def __init__(self, params):

        super().__init__(params)
        self.env = gym.make('CartPole-v0')
        self.env.reset()
        self.q_cont, self.q_frzn = self._init_models()

    # ------------------------------------------------------
    # Functions for handling the buffer (experience replay)

    def append(self, experiences):
        if len(self.buffer) + len(experiences) > self.buffer_size:
            idx = len(self.buffer) + len(experiences) - self.buffer_size
            del self.buffer[0:idx]

        self.buffer += experiences

    def _init_models(self):

        def build(strc):
            q = Sequential()
            q.add(Dense(strc[0][0], input_shape=(4,), activation=strc[0][1]))

            for i in range(1, len(strc)-1):
                q.add(Dense(strc[i][0], activation=strc[i][1]))

            q.add(Dense(2, activation=strc[-1][1]))

            optz = self._init_optimizer()
            q.compile(loss='mse', optimizer=optz)
            return q

        q_cont = build(self.net)
        q_frzn = build(self.net)

        q_cont.set_weights(q_frzn.get_weights())

        return q_cont, q_frzn


class DqnHigh(DqnBase):

    def __init__(self, params):

        super().__init__(params)
        self.env = gym.make('CartPoleRawImg-v0')
        self.env.reset()
        self.q_cont, self.q_frzn = self._init_models()

    # ------------------------------------------------------
    # Functions for handling the buffer (experience replay)
    # ------------------------------------------------------
    # format: [exp1, exp2, exp3 ...] exp1 = (obs_t, rw, action, done, obs_t+1)
    # obs_t is a numpy array with shape: (84 x 84 x 4)
    def append(self, experiences):

        def preprocess(alist):

            transformed_list = []
            for pic in alist:
                pic0 = pic[40:124, :]
                pic1 = np.zeros((pic0.shape[0], pic0.shape[1], 1))
                pic1[:, :] = pic0[:, :, 0] * 0.2126 + pic0[:, :, 1] * 0.7152 + pic0[:, :, 2] * 0.0722
                pic2 = trf.resize(pic1, (84, 84, 1))
                transformed_list.append(pic2)

            return transformed_list

        experiences = preprocess(experiences)

        pos = 0
        if len(self.buffer) + len(experiences) > self.buffer_size:
            idx = len(self.buffer) + len(experiences) - self.buffer_size
            del self.buffer[0:idx]
        elif len(self.buffer) == 0:
            if len(experiences) >= 4:
                obs = np.zeros((84, 84, 4), dtype=np.uint8)
                obs[:, :, 0] = experiences[0][0][:, :, 0]
                obs[:, :, 1] = experiences[1][0][:, :, 0]
                obs[:, :, 2] = experiences[2][0][:, :, 0]
                obs[:, :, 3] = experiences[3][0][:, :, 0]

                obs_nx = np.zeros((84, 84, 4), dtype=np.uint8)
                obs_nx[:, :, 0] = experiences[0][4][:, :, 0]
                obs_nx[:, :, 1] = experiences[1][4][:, :, 0]
                obs_nx[:, :, 2] = experiences[2][4][:, :, 0]
                obs_nx[:, :, 3] = experiences[3][4][:, :, 0]

                exp = (obs, experiences[3][1], experiences[3][2], experiences[3][3], obs_nx)
                self.buffer.append(exp)
                pos += 4 # four element is already used
            else:
                err_msg = 'There is no enough experience to create a sample!'
                self.log.log(Mode.LOG_F, 'Error: ' + err_msg)
                raise AttributeError(err_msg)

        while pos < len(experiences):
            exp = self.buffer[-1]
            obs = np.zeros((84, 84, 4), dtype=np.uint8)
            obs[:, :, 1:4] = exp[0][:, :, 0:3]
            obs[:, :, 3] = experiences[pos][0][:, :, 3]

            obs_nx = np.zeros((84, 84, 4), dtype=np.uint8)
            obs_nx[:, :, 0:3] = exp[4][:, :, 0:3]
            obs_nx[:, :, 3] = experiences[pos][3][3][:, :, 3]

            stacked_exp = (obs, experiences[pos][1], experiences[pos][2], experiences[pos][3], obs_nx)
            self.buffer.append(stacked_exp)
            pos += 1

    def _init_models(self):

        def build(strc):
            q = Sequential()
            q.add(Convolution2D(strc[0][0], kernel_size=(3, 3), padding='valid', input_shape=(84, 84, 4), activation=strc[0][1]))

            for i in range(1, len(strc)-1):  # the last should be a Dense
                q.add(Convolution2D(strc[i][0], kernel_size=(3, 3), padding='valid', activation=strc[i][1]))

            q.add(Flatten())
            q.add(Dense(2, activation=strc[-1][1]))

            optz = self._init_optimizer()
            q.compile(loss='mse', optimizer=optz)
            return q

        q_cont = build(self.net)
        q_frzn = build(self.net)

        q_cont.set_weights(q_frzn.get_weights())

        return q_cont, q_frzn
