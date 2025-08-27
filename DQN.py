# train_dqn.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pygame

# Import your custom environment
from environment import CleaningEnv

# --- 1. Q-Network Architecture ---
class QNetwork(nn.Module):
    """Defines the neural network model for the agent."""
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_size)

    def forward(self, x):
        """The forward pass of the network."""
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# --- 2. DQN Agent Class ---
class DQNAgent:
    """Manages the agent's learning process."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.update_target_every = 4

        self.policy_net = QNetwork(state_size, action_size)
        self.target_net = QNetwork(state_size, action_size)
        self.update_target_net()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.policy_net(state)
        return torch.argmax(act_values).item()

    def train_on_batch(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([e[0] for e in minibatch]))
        actions = torch.LongTensor([e[1] for e in minibatch]).unsqueeze(1)
        rewards = torch.FloatTensor([e[2] for e in minibatch]).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([e[3] for e in minibatch]))
        dones = torch.BoolTensor([e[4] for e in minibatch]).unsqueeze(1)

        current_q_values = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        
        target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    # --- NEW: Method to save the trained model's weights ---
    def save(self, filename):
        """Saves the weights of the policy network."""
        torch.save(self.policy_net.state_dict(), filename)
        print(f"Model saved to {filename}")

    # --- NEW: Method to load a pre-trained model's weights ---
    def load(self, filename):
        """Loads weights into the policy and target networks."""
        self.policy_net.load_state_dict(torch.load(filename))
        self.update_target_net() # Sync the target network
        self.epsilon = self.epsilon_min # Set epsilon to a low value for exploitation
        print(f"Model loaded from {filename}")

# --- 3. Training Function ---
def train_agent(env, agent, episodes, max_steps, render_every):
    clock = pygame.time.Clock()
    scores = []

    for e in range(episodes):
        state = env.reset()
        
        for time in range(max_steps):
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
                
                avg_score = np.mean(scores[-100:]) if len(scores) > 100 else np.mean(scores)
                print(f"Episode: {e+1}/{episodes}, Score: {time}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")
                break
            
            agent.train_on_batch()
            
    # --- NEW: Save the model after the training loop is finished ---
    agent.save("dqn_cleaning_robot.pth")
    env.close()
    print("Training finished.")

# --- 4. Main Execution Block ---
if __name__ == '__main__':
    env = CleaningEnv(grid_size=50)
    state_size = env.grid_size * env.grid_size + 2
    action_size = env.action_space
    agent = DQNAgent(state_size, action_size)

    TOTAL_EPISODES = 2000
    MAX_STEPS_PER_EPISODE = 500
    RENDER_EVERY_N_EPISODES = 50

    train_agent(
        env=env, 
        agent=agent, 
        episodes=TOTAL_EPISODES, 
        max_steps=MAX_STEPS_PER_EPISODE, 
        render_every=RENDER_EVERY_N_EPISODES
    )
