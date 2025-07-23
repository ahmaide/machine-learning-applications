from task2_1 import abstraction
from task2_2 import Env
from task2_3 import Agent


from tqdm import tqdm

def train_agent(map_file, start, goal, map_size=(80, 80), learning_rate=0.1, discount_factor=0.9,
                initial_epsilon=1.0, epsilon_decay=0.99, final_epsilon=0.1, episodes=1000, max_steps=200):

    the_map = abstraction(map_size[0], map_size[1], map_file)
    env = Env(the_map, start, goal)
    agent = Agent(env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon, discount_factor)

    for _ in tqdm(range(episodes), desc="Training"):
        agent.interact_Q_learning(1, max_steps)

    return agent

def evaluate_agent(agent, env, episodes=100, max_steps=300):
    agent.epsilon = 0.2
    success_count = 0
    total_steps = 0
    successful_steps = []

    for _ in range(episodes):
        state = env.reset()[0]
        terminated = False
        steps = 0

        while not terminated and steps < max_steps:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            steps += 1

        if terminated:
            success_count += 1
            total_steps += steps
            successful_steps.append(steps)

    success_rate = success_count / episodes
    avg_steps = (total_steps / success_count) if success_count > 0 else float('inf')

    print(f"Evaluation over {episodes} episodes:")
    print(f"Success rate: {success_rate}")
    print(f"Average Steps to Goal, for successful episodes: {avg_steps}")

    return success_rate, avg_steps


trained_agent = train_agent(
        map_file='map1.bmp',
        start=(0, 0),
        goal=(45, 45),
        map_size=(80, 80),
        learning_rate=0.2,
        discount_factor=0.95 ,
        initial_epsilon=1.0,
        epsilon_decay=0.995,
        final_epsilon=0.05,
        episodes=10000,
        max_steps=300
    )

env = trained_agent.env

evaluate_agent(trained_agent, env, episodes=100, max_steps=300)