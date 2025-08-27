
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
        
        self.action_space = 4
        self.grid = None
        self.dust_count = 0
        
        # --- NEW: Define rewards as configurable attributes ---
        self.reward_finished = 100.0
        self.reward_crash = -100.0
        self.reward_clean = 10.0
        self.reward_time_step = -0.1

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.robot.reset()
        
        self._place_random_objects(object_type='obstacle', min_size=3, max_size=8, count=7)
        self._place_random_objects(object_type='dust', min_size=2, max_size=4, count=15)
        self.dust_count = np.sum(self.grid == 2)

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

                if np.sum(self.grid[y:y+height, x:x+width]) == 0:
                    self.grid[y:y+height, x:x+width] = grid_value
                    break
    
    def step(self, action):
        self.robot.move(action)
        
        # Start with a small time penalty for every action
        reward = self.reward_time_step
        done = False
        
        # --- UPDATED: Explicit flags for success and failure ---
        
        # Get coordinates and tiles under the robot's body
        body_coords = self.robot.get_body_coords()
        rows = np.array([coord[0] for coord in body_coords])
        cols = np.array([coord[1] for coord in body_coords])
        tiles_under_robot = self.grid[rows, cols]
        
        # 1. Check for FAILURE state (collision)
        if np.any(tiles_under_robot == 1):
            reward = self.reward_crash # Assign failure reward
            done = True
        
        # 2. If not failed, perform actions and check for SUCCESS
        else:
            # Clean dust automatically
            dust_mask = (tiles_under_robot == 2)
            num_dust_cleaned = np.sum(dust_mask)
            
            if num_dust_cleaned > 0:
                reward += self.reward_clean * num_dust_cleaned 
                cleaned_rows = rows[dust_mask]
                cleaned_cols = cols[dust_mask]
                self.grid[cleaned_rows, cleaned_cols] = 0
                self.dust_count -= num_dust_cleaned

            # Check for SUCCESS state (all dust cleaned)
            if self.dust_count <= 0:
                reward = self.reward_finished # Assign success reward
                done = True

        # Construct the next state for the agent
        robot_pos_normalized = np.array(self.robot.pos) / self.grid_size
        next_state = np.concatenate([self.grid.flatten(), robot_pos_normalized])

        return next_state, reward, done

    def render(self):
        self.renderer.render(self.grid, self.dust_count, self.robot)

    def close(self):
        self.renderer.close()