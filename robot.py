class Robot:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.start_pos = [self.grid_size - 1, 0] # Bottom-left corner
        self.pos = list(self.start_pos) # Use list() to create a copy

    def reset(self):
        """Resets the robot to its starting position."""
        self.pos = list(self.start_pos)

    def move(self, action):
        """
        Updates the robot's position based on an action.
        Action: 0:Up, 1:Down, 2:Left, 3:Right.
        """
        if action == 0: # Up
            self.pos[0] = max(0, self.pos[0] - 1)
        elif action == 1: # Down
            self.pos[0] = min(self.grid_size - 1, self.pos[0] + 1)
        elif action == 2: # Left
            self.pos[1] = max(0, self.pos[1] - 1)
        elif action == 3: # Right
            self.pos[1] = min(self.grid_size - 1, self.pos[1] + 1)