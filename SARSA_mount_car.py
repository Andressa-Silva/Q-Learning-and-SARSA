#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

class MountainCarSARSA:
    def __init__(self, episodes, is_training=True, render=False):
        self.episodes = episodes
        self.is_training = is_training
        self.render = render
        self.env = gym.make('MountainCar-v0', render_mode='human' if render else None)

        #Divide position and velocity into segments
        self.pos_space = np.linspace(self.env.observation_space.low[0], self.env.observation_space.high[0], 20) #Between -1.2 and 0.6
        self.vel_space = np.linspace(self.env.observation_space.low[1], self.env.observation_space.high[1], 20) #Between -0.07 and 0.07

        if self.is_training:
            self.q = np.zeros((len(self.pos_space), len(self.vel_space), self.env.action_space.n)) #Init a 20x20x3 array
        else:
            with open('SARSA_mountain_car.pkl', 'rb') as f:
                self.q = pickle.load(f)

        self.learning_rate_a = 0.1            #Alpha or learning rate
        self.discount_factor_g = 0.99         #Gamma or discount factor.
        self.epsilon = 1                      #1 = 100% random actions
        self.epsilon_decay_rate = 2 / episodes
        self.rng = np.random.default_rng()    #Random number generator
        self.rewards_per_episode = np.zeros(episodes)

    def run(self):
        for i in range(self.episodes):
            state = self.env.reset()[0] #Starting position, starting velocity always 0
            state_p = np.digitize(state[0], self.pos_space)
            state_v = np.digitize(state[1], self.vel_space)

            terminated = False
            rewards = 0

            if self.is_training and self.rng.random() < self.epsilon:
                #Choose random action (0=drive left, 1=stay neutral, 2=drive right)
                action = self.env.action_space.sample()
            else:
                action = np.argmax(self.q[state_p, state_v, :])

            while not terminated and rewards > -1000: #Reward of -1 for each timestep
                new_state, reward, terminated, _, _ = self.env.step(action)
                new_state_p = np.digitize(new_state[0], self.pos_space)
                new_state_v = np.digitize(new_state[1], self.vel_space)

                if self.is_training and self.rng.random() < self.epsilon:
                    new_action = self.env.action_space.sample()
                else:
                    new_action = np.argmax(self.q[new_state_p, new_state_v, :])

                if self.is_training:
                    self.q[state_p, state_v, action] += self.learning_rate_a * (
                        reward + self.discount_factor_g * self.q[new_state_p, new_state_v, new_action] - self.q[state_p, state_v, action]
                    )

                state = new_state
                state_p = new_state_p
                state_v = new_state_v
                action = new_action

                rewards += reward
                
                if not self.is_training and rewards % 100 == 0:
                    print(f'Episode: {i}  Rewards: {rewards}')
                    np.set_printoptions(threshold=sys.maxsize)
                    print(self.q)

                if self.is_training and rewards % 100 == 0:
                    print(f'Episode: {i}  Rewards: {rewards}')

            self.epsilon = max(self.epsilon - self.epsilon_decay_rate, 0)
            self.rewards_per_episode[i] = rewards

        self.env.close()

        #Save Q table to file
        if self.is_training:
            with open('SARSA_mountain_car.pkl', 'wb') as f:
                pickle.dump(self.q, f)

        self.plot_rewards()

    def plot_rewards(self):
        mean_rewards = np.zeros(self.episodes)
        for t in range(self.episodes):
            mean_rewards[t] = np.mean(self.rewards_per_episode[max(0, t-100):(t+1)])
        plt.plot(mean_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.title('SARSA Mountain Car')
        plt.savefig('mountain_car_sarsa.png')

if __name__ == '__main__':
    # Treinamento
    agent = MountainCarSARSA(5000, is_training=True, render=False)
    agent.run()

    # Teste
    agent = MountainCarSARSA(1, is_training=False, render=True)
    agent.run()
