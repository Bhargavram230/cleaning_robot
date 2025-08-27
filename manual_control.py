# main.py

import pygame
import sys

# Assuming your other classes are in their respective files
from environment import CleaningEnv

def manual_control():
    """
    Initializes the environment and runs the main loop for manual control.
    """
    env = CleaningEnv(grid_size=50)
    state = env.reset()
    
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = None # No action by default
        
        # --- Keyboard Input Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    action = 0 # Up
                elif event.key == pygame.K_s:
                    action = 1 # Down
                elif event.key == pygame.K_a:
                    action = 2 # Left
                elif event.key == pygame.K_d:
                    action = 3 # Right
                elif event.key == pygame.K_SPACE:
                    action = 4 # Stay / "Suck" action
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # If a valid key was pressed, take a step in the environment
        if action is not None:
            next_state, reward, done = env.step(action)
            print(f"Action: {action}, Reward: {reward:.1f}, Done: {done}")
            
            # If the episode is over, reset the environment
            if done:
                print("Episode finished! Resetting...")
                env.render()
                pygame.time.wait(1000) # Pause for 1 second
                state = env.reset()

        # Render the current state of the environment
        env.render()
        
        # Control the frame rate
        clock.tick(30)
            
    env.close()
    print("Game exited.")

if __name__ == '__main__':
    manual_control()
