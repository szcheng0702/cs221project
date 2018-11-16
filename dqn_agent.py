# This code is largely based on the Deep Q-Learning template published at:
# https://github.com/keon/deep-q-learning.git

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 1000
TIME_LIMIT = 1000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000) # maximum number of samples stored in dataset
        self.gamma = 0.95 # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01 # minimum exploration rate
        self.epsilon_decay = 0.995 # decay rate for exploration
        self.learning_rate = 0.001
        self.model = self._build_model_3L()

    def _build_model_2L(self):
        """2-layer Neural Net for Deep-Q learning Model."""
        model = Sequential()
        model.add(Dense(units=24, input_dim=self.state_size, activation='relu')) # input layer
        model.add(Dense(units=self.action_size, activation='linear')) # output layer
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate)) # loss function = mean squared error
        return model
    
    def _build_model_3L(self):
        """3-layer Neural Net for Deep-Q learning Model."""
        model = Sequential()
        model.add(Dense(units=24, input_dim=self.state_size, activation='relu')) # input layer
        model.add(Dense(units=24, activation='relu'))
        model.add(Dense(units=self.action_size, activation='linear')) # output layer
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate)) # loss function = mean squared error
        return model

    def _build_model_4L(self):
        """4-layer Neural Net for Deep-Q learning Model."""
        model = Sequential()
        model.add(Dense(units=24, input_dim=self.state_size, activation='relu')) # input layer
        model.add(Dense(units=24, activation='relu'))
        model.add(Dense(units=24, activation='relu'))
        model.add(Dense(units=self.action_size, activation='linear')) # output layer
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate)) # loss function = mean squared error
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store s,a,r,s' by appending to self.memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action randomly (explore) or by model prediction (exploit)."""
        if np.random.rand() <= self.epsilon: # explore with probabiluty self.epsilon
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        """Train the neural net on the episodes in self.memory. 
           Only N samples defined by batch_size are sampled from self.memory for training.
        """
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) # epochs = number of iterations over the minibatch

        # Decaying exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == '__main__':
    env = gym.make('Phoenix-ram-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 500

    for episode in range(EPISODES):
        print('episode = {}'.format(episode))
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(TIME_LIMIT):
            # env.render()
            action = agent.act(state) # DQN agent chooses next action 
            next_state, reward, done, _ = env.step(action) # observe rewards and successor state
            # reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done) # add s,a,r,s' to dataset (agent.memory)
            state = next_state

            if done:
                print('episode: {}/{}, score: {}, exploration rate: {:.2}'
                      .format(episode, EPISODES, time, agent.epsilon))
                break

        # Train NN after each episode or timeout by randomly sampling a batch from the dataset in agent.memory
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    
    # Save weights after training is complete
    agent.save('./save/phoenix_dqn_3L.h5')
