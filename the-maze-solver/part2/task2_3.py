import numpy as np
import gymnasium as gym
from task2_2 import Env
from task2_1 import abstraction

class Agent:
    def __init__(self, env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon, discount_factor):
        self.env = env
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

        y, x = env.map.shape
        action_num = env.action_space.n
        self.q_val = np.zeros((y, x, action_num))

    def get_action(self, state):
        y, x = state
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_val[y, x]))

    def update(self, state, action, reward, terminated, next_state):
        y, x = state
        ny, nx = next_state
        new_q = 0 if terminated else np.max(self.q_val[ny, nx])
        td_error = reward + self.discount_factor * new_q - self.q_val[y, x, action]
        self.q_val[y, x, action] += self.lr * td_error
        self.training_error.append(td_error)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def interact(self, episodes):
        success = False
        for e in range(episodes):
            state = self.env.reset()[0]
            terminated = False
            
            while not terminated:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.update(state, action, reward, terminated, next_state)
                state = next_state
            
            #print("terminated: ", terminated)

            self.decay_epsilon()
            if terminated:
                success = True

        return success

    # For task 2.4
    def interact_Q_learning(self, episodes, max_steps):
        success_count = 0
        for e in range(episodes):
            state = self.env.reset()[0]
            terminated = False
            steps = 0

            while not terminated and steps < max_steps:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.update(state, action, reward, terminated, next_state)
                state = next_state
                steps += 1

            self.decay_epsilon()
            if terminated:
                success_count += 1

        return success_count



################## testing ########################

'''
def testing():

    a_map = abstraction(80, 80, 'map1.bmp')
    #a_map = abstraction(80, 80, 'map2.bmp')
    env = Env(a_map, (0, 0), (59, 59))
    agent = Agent(env, 0.1, 1.0, 0.99, 0.1, 0.9)

    success = agent.interact(episodes=1)

    if success == True:
        print('successful')

    else:
        print("not successful")

testing()
'''
