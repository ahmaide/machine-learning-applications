import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import trange
from collections import namedtuple

Transition = namedtuple( 'Transition', ('state', 'action', 'reward', 'next_s', 'done'))

__name__ = "__main__"

np.random.seed(58)
disc = 101

class ODEModelEnv:
    def __init__(self, disc_scheme, steps):
        self.state = None

        self.x_range = [-5, 5]
        self.v_range = [-5, 5]

        self.X = np.linspace(-5, 5, disc_scheme)
        self.V = np.linspace(-5, 5, disc_scheme)

        self.control_inputs = np.array([-5, -1, -0.1, -0.01, -0.001, -0.0001, 0,
                                        0.0001, 0.001, 0.01, 0.1, 1, 5])
        self.delta = 0.1
        self.max_steps = steps

        self.steps = 0
        self.reset()

    def reset(self, initial_state=None):
        self.steps = 0

        if initial_state is None:
            x = np.random.uniform(-5, 5)
            v = np.random.uniform(-5, 5)
            self.state = np.array([x, v])
        
        else:
            self.state = np.array(initial_state)
        
        return self.discretize(self.state)

    def discretize(self, state):
        x = max(min(state[0], self.x_range[1]), self.x_range[0])
        v = max(min(state[1], self.v_range[1]), self.v_range[0])
        
        x_i = np.argmin(np.abs(self.X - x))
        v_i = np.argmin(np.abs(self.V - v))
        
        return (x_i, v_i)

    def step(self, action_idx):
        u = self.control_inputs[action_idx]
        t = np.linspace(0, self.delta)

        def model(s, t, u):
            return [s[1], u]

        y = odeint(model, self.state, t, args=(u,))[-1]
        self.state = y

        distance = np.linalg.norm(self.state)
        if self.steps >= self.max_steps or distance < 0.01:
            done = True
        
        else:
            done = False

        if distance < 0.01:
            reward = 100

        else:
            reward = -distance         
            reward -= 0.1 * abs(u)                  
            reward -= 0.01                              

        if distance >= 0.01 and done:
            reward -= 10

        if abs(y[0]) < 0.05 and abs(y[1]) < 0.05:
            reward += 12

        s = self.state
        self.steps += 1

        return self.discretize(y), reward, done, s

class Agent:
    def __init__(self, 
                learning_rate=0.1, 
                discount_factor=0.90,
                epsilon_greedy=1.0, 
                epsilon_min=0.05, 
                epsilon_decay=0.995):

        self.q_table = np.zeros((disc, disc, 13))
        self.control_inputs = np.array([-5, -1, -0.1, -0.01, -0.001, -0.0001, 0,
                                        0.0001, 0.001, 0.01, 0.1, 1, 5])
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(len(self.control_inputs))

        else:
            action = np.argmax(self.q_table[state])
        
        return action

    def _learn(self, transition):
        s, a, r, next_s, done = transition
        q_val = self.q_table[s][a]

        if done:
            q_target = r
        
        else:
            q_target = r + self.gamma * np.max(self.q_table[next_s])
        
        self.q_table[s][a] += self.lr * (q_target - q_val)
        self._adjust_epsilon()
    
    def _adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def run_qlearning(agent, env, num_episodes):
    for _ in trange(num_episodes, desc="Training"):
        state = env.reset()
        t_reward = 0

        while True:
            action = agent.choose_action(state)
            next_s, reward, done, _ = env.step(action)
            agent._learn(Transition(state, action, reward, next_s, done))
            state = next_s
            t_reward += reward
            
            if done:
                break

def generate_trajectory(agent, env, initial_state=None):
    state = env.reset(initial_state)
    
    t = []
    x = []
    v = []

    time = 0
    for _ in range(200):
        action = np.argmax(agent.q_table[state])
        next_s, _, done, s = env.step(action)

        time += env.delta
        t.append(time)
        x.append(s[0])
        v.append(s[1])
        
        state = next_s
    total = 0
    for i in x:
        total += abs(i)
    print("Average Distance")
    print(total/200)
    plt.figure(figsize=(10, 6))
    plt.plot(t, x, 'r-')
    plt.plot(t, v, 'b-')
    plt.xlabel('t')
    plt.show()

if __name__ == '__main__':
    env = ODEModelEnv(disc, steps = 500)
    agent = Agent(learning_rate=0.1, discount_factor=0.99, epsilon_greedy=1.0, epsilon_min=0.01, epsilon_decay=0.9995)

    run_qlearning(agent, env, num_episodes=10000)
    generate_trajectory(agent, env, [1, -1])
    

