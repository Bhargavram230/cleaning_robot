# robot.py

class Robot:
    def __init__(self, grid_size, size=3):
        self.grid_size = grid_size
        self.size = size
        self.radius = self.size // 2 # For a 3x3 robot, radius is 1
        
        # CORRECTED: The bottom-left-most valid center position
        self.start_pos = [self.grid_size - 1 - self.radius, self.radius]
        self.pos = list(self.start_pos)

    def reset(self):
        """Resets the robot to its starting position."""
        self.pos = list(self.start_pos)

    def move(self, action):
        """Updates the robot's center position based on an action."""
        min_coord = self.radius
        max_coord = self.grid_size - 1 - self.radius
        
        if action == 0: # Up
            self.pos[0] = max(min_coord, self.pos[0] - 1)
        elif action == 1: # Down
            self.pos[0] = min(max_coord, self.pos[0] + 1)
        elif action == 2: # Left
            self.pos[1] = max(min_coord, self.pos[1] - 1)
        elif action == 3: # Right
            self.pos[1] = min(max_coord, self.pos[1] + 1)

    def get_body_coords(self):
        """Gets the coordinates of all cells occupied by the robot."""
        center_r, center_c = self.pos
        coords = []
        for r in range(center_r - self.radius, center_r + self.radius + 1):
            for c in range(center_c - self.radius, center_c + self.radius + 1):
                coords.append((r, c))
        return coords
