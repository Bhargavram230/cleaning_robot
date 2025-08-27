import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- 1. Define the Q-Network Architecture in a Separate Class ---
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        
        # Hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # Create the policy and target networks from the QNetwork class
        self.policy_net = QNetwork(state_size, action_size)
        self.target_net = QNetwork(state_size, action_size)
        self.update_target_net()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def update_target_net(self):
        """Copies weights from the policy network to the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience in the replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Chooses an action using the epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.policy_net(state)
        return torch.argmax(act_values).item()

    def train_on_batch(self):

        if len(self.memory) < self.batch_size:
            return
        
        # 1. Sample a random batch of experiences
        minibatch = random.sample(self.memory, self.batch_size)
        
        # 2. Unpack the batch into separate Tensors (this is a vectorized operation)
        states = torch.FloatTensor(np.array([e[0] for e in minibatch]))
        actions = torch.LongTensor([e[1] for e in minibatch]).unsqueeze(1)
        rewards = torch.FloatTensor([e[2] for e in minibatch]).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([e[3] for e in minibatch]))
        dones = torch.BoolTensor([e[4] for e in minibatch]).unsqueeze(1)

        # 3. Get the Q-values for the current states from the policy network
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # 4. Get the maximum Q-values for the next states from the target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        
        # 5. Compute the target Q-values using the Bellman equation
        target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        # 6. Calculate the loss and update the policy network
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 7. Decay epsilon for exploration vs. exploitation trade-off
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay