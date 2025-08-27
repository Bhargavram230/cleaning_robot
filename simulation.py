import pygame
import sys

class PygameRenderer:
    def __init__(self, grid_size, screen_size=600):
        pygame.init()
        self.screen_size = screen_size
        self.grid_size = grid_size
        self.cell_size = self.screen_size // self.grid_size
        
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Cleaning Robot Simulation")
        
        self.colors = {
            0: (255, 255, 255), # Empty
            1: (40, 40, 40),     # Obstacle
            2: (255, 215, 0),   # Dust
            'robot': (220, 20, 60) # Robot Color
        }

    def render(self, grid, dust_count, robot):
        """ Renders the grid and then the robot on top. """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(self.colors[0])
        
        # Step 1: Draw the static environment (obstacles and dust)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell_value = grid[r, c]
                if cell_value != 0:
                    color = self.colors.get(cell_value)
                    rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, color, rect)

        # Step 2: Draw the robot on top of the environment
        robot_color = self.colors['robot']
        for r, c in robot.get_body_coords():
            rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, robot_color, rect)

        pygame.display.set_caption(f"Cleaning Robot - Dust Remaining: {dust_count}")
        pygame.display.flip()

    def close(self):
        pygame.quit()