# train_dqn.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pygame

# Import your custom environment from its file
from environment import CleaningEnv
from DQN import DQNAgent



# --- 3. NEW: Training Function ---
def train_agent(env, agent, episodes, max_steps, render_every):
    """
    Main function to train the DQN agent.

    Args:
        env: The cleaning environment instance.
        agent: The DQN agent instance.
        episodes: The total number of episodes to train for.
        max_steps: The maximum number of steps per episode.
        render_every: The frequency (in episodes) to render the environment.
    """
    clock = pygame.time.Clock()
    scores = []

    for e in range(episodes):
        state = env.reset()
        
        for time in range(max_steps):
            # Render the environment to visualize progress
            if e > 0 and e % render_every == 0:
                env.render()
                clock.tick(30)

            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                scores.append(time)
                if e % agent.update_target_every == 0:
                    agent.update_target_net()
                
                avg_score = np.mean(scores[-10:]) if len(scores) > 10 else np.mean(scores)
                print(f"Episode: {e+1}/{episodes}, Score: {time}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")
                break
            
            agent.train_on_batch()
            
    env.close()
    print("Training finished.")

# --- 4. Main Execution Block ---
if __name__ == '__main__':
    # Initialize Environment and Agent
    env = CleaningEnv(grid_size=50)
    state_size = env.grid_size * env.grid_size + 2
    action_size = env.action_space
    agent = DQNAgent(state_size, action_size)

    # Define Training Parameters
    TOTAL_EPISODES = 2000
    MAX_STEPS_PER_EPISODE = 500
    RENDER_EVERY_N_EPISODES = 50

    # Start Training
    train_agent(
        env=env, 
        agent=agent, 
        episodes=TOTAL_EPISODES, 
        max_steps=MAX_STEPS_PER_EPISODE, 
        render_every=RENDER_EVERY_N_EPISODES
    )
