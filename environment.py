import numpy as np
import random
import matplotlib.pyplot as plt

from robot import Robot

class CleaningEnv:
    def __init__(self, grid_size=200):
        self.grid_size = grid_size
        self.robot = Robot(self.grid_size) 
        
        # Action space: 0:Up, 1:Down, 2:Left, 3:Right, 4:Suck Dust
        self.action_space = 5
        self.state_space = self.grid_size * self.grid_size
        self.grid = None
        self.dust_count = 0

    def reset(self):
        """Resets the environment for a new episode."""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.robot.reset()
        
        self._place_random_objects(object_type=1, min_size=5, max_size=20, count=10)
        
        self.dust_count = 20
        self._place_random_objects(object_type=2, min_size=2, max_size=5, count=self.dust_count)
        
        self._update_robot_on_grid()
        return self.grid.flatten()

    def _place_random_objects(self, object_type, min_size, max_size, count):
        for _ in range(count):
            while True:
                width = random.randint(min_size, max_size)
                height = random.randint(min_size, max_size)
                x = random.randint(0, self.grid_size - width)
                y = random.randint(0, self.grid_size - height)

                if np.sum(self.grid[y:y+height, x:x+width]) == 0:
                    self.grid[y:y+height, x:x+width] = object_type
                    break
    
    def _update_robot_on_grid(self):
        """Updates the grid to show the robot's current position."""
        # This function assumes the previous robot position is already cleared
        self.grid[self.robot.pos[0], self.robot.pos[1]] = 3 # 3 represents the robot

    def step(self, action):
        """Executes one time step in the environment."""
        old_pos = list(self.robot.pos)
        
        # Erase the robot's old position from the grid
        # Only erase if it was the robot, not if it was on top of dust
        if self.grid[old_pos[0], old_pos[1]] == 3:
             self.grid[old_pos[0], old_pos[1]] = 0 # Set to empty space
        
        # --- Action Logic ---
        # Actions 0-3 are movement actions
        if action < 4:
            self.robot.move(action)
        
        # --- Reward & State Logic ---
        current_tile_value = self.grid[self.robot.pos[0], self.robot.pos[1]]
        reward = 0
        
        if current_tile_value == 1: # Collided with an obstacle
            reward = -10
        elif action == 4: # Suck Dust
            if current_tile_value == 2: # Successfully cleaned dust
                reward = 20
                self.dust_count -= 1
                # The spot is now empty, it will be updated with the robot's position later
            else: # Tried to suck on a non-dust spot
                reward = -1
        else: # Moved without incident
            reward = -0.1

        # Update grid with the robot's new position
        self._update_robot_on_grid()

        # --- Done Condition ---
        done = self.dust_count == 0
        if done:
            reward += 100 # Bonus for finishing

        return self.grid.flatten(), reward, done

    def render(self, pause_time=0.1):
        """Renders the environment."""
        plt.imshow(self.grid, cmap='viridis')
        plt.title(f"Dust Remaining: {self.dust_count}")
        plt.show(block=False)
        plt.pause(pause_time)
        plt.clf() # Clear the figure for the next frame

# --- Example Usage ---
if __name__ == '__main__':
    env = CleaningEnv()
    state = env.reset()
    plt.figure() # Create a figure window once

    # Simulate a few random steps
    for i in range(50):
        action = random.randint(0, 4) # Choose a random action
        next_state, reward, done = env.step(action)
        print(f"Step: {i+1}, Action: {action}, Reward: {reward:.1f}, Done: {done}")
        env.render()
        if done:
            print("Cleaning complete!")
            break
    
    plt.close() # Close the plot window at the end