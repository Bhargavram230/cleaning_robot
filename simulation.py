class PygameRenderer:
    def __init__(self, grid_size, screen_size=600):
        pygame.init()
        self.screen_size = screen_size
        self.grid_size = grid_size
        self.cell_size = self.screen_size // self.grid_size
        
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Cleaning Robot Simulation")
        
        # Define colors
        self.colors = {
            0: (255, 255, 255), # Empty (White)
            1: (40, 40, 40),     # Obstacle (Dark Gray)
            2: (255, 215, 0),   # Dust (Gold)
            3: (220, 20, 60)    # Robot (Crimson)
        }

    def render(self, grid, dust_count):
        # Handle Pygame events, like closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(self.colors[0]) # Fill background with empty color
        
        # Draw each cell of the grid
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell_value = grid[r, c]
                if cell_value != 0: # Only draw non-empty cells for efficiency
                    color = self.colors.get(cell_value, (0, 0, 0)) # Default to black if unknown
                    rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, color, rect)

        # Update the display caption with the dust count
        pygame.display.set_caption(f"Cleaning Robot - Dust Remaining: {dust_count}")
        pygame.display.flip() # Update the full screen

    def close(self):
        pygame.quit()
