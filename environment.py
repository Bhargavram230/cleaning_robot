
import numpy as np
import random
import pygame
from robot import Robot
from simulation import PygameRenderer
class CleaningEnv:
    def __init__(self, grid_size=50):
        self.grid_size = grid_size
        self.robot = Robot(self.grid_size, size=3)
        self.renderer = PygameRenderer(self.grid_size)
        
        self.action_space = 5
        self.grid = None
        self.dust_count = 0
        self.obstacles = []
        self.dust_patches = []

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.robot.reset()
        
        self.obstacles.clear()
        self.dust_patches.clear()
        
        self._place_random_objects(object_type='obstacle', min_size=3, max_size=8, count=7)
        self.dust_count = 15
        self._place_random_objects(object_type='dust', min_size=2, max_size=4, count=self.dust_count)
        
        robot_pos_normalized = np.array(self.robot.pos) / self.grid_size
        state = np.concatenate([self.grid.flatten(), robot_pos_normalized])
        return state

    def _place_random_objects(self, object_type, min_size, max_size, count):
        grid_value = 1 if object_type == 'obstacle' else 2
        for _ in range(count):
            while True:
                width = random.randint(min_size, max_size)
                height = random.randint(min_size, max_size)
                x = random.randint(0, self.grid_size - width)
                y = random.randint(0, self.grid_size - height)

                robot_start_coords = self.robot.get_body_coords()
                
                is_overlapping = False
                for r_start, c_start in robot_start_coords:
                    if y <= r_start < y + height and x <= c_start < x + width:
                        is_overlapping = True
                        break
                
                if not is_overlapping and np.sum(self.grid[y:y+height, x:x+width]) == 0:
                    self.grid[y:y+height, x:x+width] = grid_value
                    obj_info = {'x': x, 'y': y, 'width': width, 'height': height}
                    if object_type == 'obstacle': self.obstacles.append(obj_info)
                    else: self.dust_patches.append(obj_info)
                    break
    
    def step(self, action):
        done = False
        # Action 4 is now a "stay" action, actions 0-3 are movement
        if action < 4:
            self.robot.move(action)
        
        # Start with a small time penalty for every action taken
        reward = -0.1
        
        # Get coordinates and tiles under the robot's body
        body_coords = self.robot.get_body_coords()
        rows = np.array([coord[0] for coord in body_coords])
        cols = np.array([coord[1] for coord in body_coords])
        tiles_under_robot = self.grid[rows, cols]
        
        # 1. Check for collision with an obstacle
        if np.any(tiles_under_robot == 1):
            reward = -100 # Overwrite the small penalty with a large one
            done = True
        
        # 2. If no collision, check for and clean dust AUTOMATICALLY
        if not done:
            # Find which of the tiles under the robot are dust (value 2)
            dust_mask = (tiles_under_robot == 2)
            num_dust_cleaned = np.sum(dust_mask)
            
            if num_dust_cleaned > 0:
                # Add a positive reward proportional to how much was cleaned
                reward += 10 * num_dust_cleaned 
                
                # Get the actual grid coordinates of the cleaned dust
                cleaned_rows = rows[dust_mask]
                cleaned_cols = cols[dust_mask]
                
                # Set those grid cells to 0 (empty)
                self.grid[cleaned_rows, cleaned_cols] = 0
                self.dust_count -= num_dust_cleaned

        # 3. Check if the episode is finished
        if self.dust_count == 0:
            done = True
            reward += 100

        # 4. Construct the next state for the agent
        robot_pos_normalized = np.array(self.robot.pos) / self.grid_size
        next_state = np.concatenate([self.grid.flatten(), robot_pos_normalized])

        return next_state, reward, done

    def render(self):
        self.renderer.render(self.grid, self.dust_count, self.robot)

    def close(self):
        self.renderer.close()


# --- Example Usage (main.py) ---
if __name__ == '__main__':
    import pygame
    
    env = CleaningEnv(grid_size=50)
    state = env.reset()
    
    clock = pygame.time.Clock()
    
    for i in range(500):
        action = random.randint(0, 4)
        next_state, reward, done = env.step(action)
        
        env.render()
        
        clock.tick(10) # Control FPS
        
        if done:
            print(f"Episode finished after {i+1} steps!")
            env.render()
            pygame.time.wait(2000)
            break
            
    env.close()
