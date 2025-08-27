import numpy as np
import random
import matplotlib.pyplot as plt

from robot import Robot
from simulation import PygameRenderer   

class CleaningEnv:
    def __init__(self, grid_size=200):
        self.grid_size = grid_size
        self.robot = Robot(self.grid_size, size=3) # Robot is now 3x3
        
        self.action_space = 5
        self.grid = None
        self.obstacles = []
        self.dust_patches = []
        self.dust_count = 0

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.robot.reset()
        
        self.obstacles.clear()
        self.dust_patches.clear()
        
        self._place_random_objects(object_type='obstacle', min_size=5, max_size=20, count=10)
        self.dust_count = 20
        self._place_random_objects(object_type='dust', min_size=2, max_size=5, count=self.dust_count)
        
        self._update_robot_on_grid(is_reset=True)
        return self.grid.flatten()

    def _place_random_objects(self, object_type, min_size, max_size, count):
        grid_value = 1 if object_type == 'obstacle' else 2
        for _ in range(count):
            while True:
                width = random.randint(min_size, max_size)
                height = random.randint(min_size, max_size)
                x = random.randint(0, self.grid_size - width)
                y = random.randint(0, self.grid_size - height)

                # Make sure we don't place objects on the robot's start area
                start_coords = self.robot.get_body_coords()
                start_area = self.grid[start_coords[0][0]:start_coords[-1][0]+1, start_coords[0][1]:start_coords[-1][1]+1]

                if np.sum(self.grid[y:y+height, x:x+width]) == 0 and np.sum(start_area) == 0:
                    self.grid[y:y+height, x:x+width] = grid_value
                    obj_info = {'x': x, 'y': y, 'width': width, 'height': height}
                    if object_type == 'obstacle':
                        self.obstacles.append(obj_info)
                    else:
                        self.dust_patches.append(obj_info)
                    break
    
    def _clear_robot_on_grid(self):
        """Clears the robot's previous position."""
        for r, c in self.robot.get_body_coords():
            # Be careful not to erase dust the robot is on top of
            if self.grid[r, c] == 3:
                self.grid[r, c] = 0

    def _update_robot_on_grid(self, is_reset=False):
        """Updates the grid to show the robot's current 3x3 body."""
        if not is_reset:
             self._clear_robot_on_grid()
        for r, c in self.robot.get_body_coords():
            self.grid[r, c] = 3

    def step(self, action):
        done = False
        
        if action < 4:
            self.robot.move(action)

        # --- NEW: Collision and Reward Logic ---
        reward = 0
        body_coords = self.robot.get_body_coords()
        
        # Check for collision
        for r, c in body_coords:
            if self.grid[r, c] == 1: # Collided with an obstacle
                reward = -100  # Large penalty
                done = True # END THE EPISODE
                break # No need to check further
        
        if not done:
            if action == 4: # Suck Dust
                center_r, center_c = self.robot.pos
                if self.grid[center_r, center_c] == 2: # Cleaned dust at center
                    reward = 20
                    self.dust_count -= 1
                    self.grid[center_r, center_c] = 0 # Remove the dust
                else:
                    reward = -1
            else: # Moved without incident
                reward = -0.1
        
        self._update_robot_on_grid()

        if self.dust_count == 0:
            done = True
            reward += 100

        return self.grid.flatten(), reward, done

    def render(self, pause_time=0.1):
        plt.imshow(self.grid, cmap='viridis')
        plt.title(f"Dust Remaining: {self.dust_count}")
        plt.show(block=False)
        plt.pause(pause_time)
        plt.clf()

# --- Example Usage ---
if __name__ == '__main__':
    env = CleaningEnv()
    state = env.reset()
    plt.figure()

    # Simulate until the episode ends
    for i in range(100): # Max 100 steps
        action = random.randint(0, 4) # Choose a random action
        next_state, reward, done = env.step(action)
        print(f"Step: {i+1}, Action: {action}, Reward: {reward:.1f}, Done: {done}")
        env.render()
        if done:
            print("Episode finished!")
            break
            
    plt.close()