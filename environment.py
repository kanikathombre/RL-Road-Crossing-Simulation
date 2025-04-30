import numpy as np
import random

class RoadCrossingEnv:
    def __init__(self, grid_size=6):
        # Initialize the environment with a square grid
        self.grid_size = grid_size
        self.reset() # Set initial state

    def reset(self):
        # Reset the environment to its initial state
        self.pedestrian_pos = [self.grid_size - 1, 0] # Start at bottom-left corner
        self.vehicles = [[i, self.grid_size - 1 - i] for i in range(self.grid_size)] # Vehicles on each row
        self.traffic_lights = [1 if i % 2 == 0 else 0 for i in range(self.grid_size)]  # Even columns have green, odd red
        self.traffic_timers = [0] * self.grid_size # Track how long each light has been in its state
        self.traffic_cycle = 3  # Number of steps before light toggles (green ↔ red)
        self.step_count = 0 # Track total steps
        return self.get_state()

    def update_traffic_lights(self):
        # Toggle each traffic light based on a timer
        for i in range(self.grid_size):
            self.traffic_timers[i] += 1 # Increment timer
            if self.traffic_timers[i] >= self.traffic_cycle:
                self.traffic_lights[i] = 1 - self.traffic_lights[i]  # Switch red ↔ green
                self.traffic_timers[i] = 0 # Reset timer

    def move_vehicles(self):
         # Move vehicles based on traffic light state
        new_vehicles = []
        for i, (r, c) in enumerate(self.vehicles):
            if self.traffic_lights[c] == 1:   # Green light → vehicle can move
                new_c = c - 1 # Move left
                if new_c >= 0:
                    new_vehicles.append([r, new_c])
                else:
                    new_vehicles.append([r, self.grid_size - 1])  # Wrap to right edge
            else:
                new_vehicles.append([r, c])  # Red light → stop
        self.vehicles = new_vehicles

    def step(self, action):
        # Perform one step in the environment based on the agent's action
        
        # Move pedestrian
        row, col = self.pedestrian_pos
        if action == 0 and row > 0:
            row -= 1  # move up
        elif action == 1 and row < self.grid_size - 1:
            row += 1  # move down
        elif action == 2 and col > 0:
            col -= 1  # move left
        elif action == 3 and col < self.grid_size - 1:
            col += 1  # move right
        self.pedestrian_pos = [row, col] # Update position

        # Update traffic lights and vehicles
        self.update_traffic_lights()
        self.move_vehicles()

        # Check for collision
        reward = -1 if self.pedestrian_pos in self.vehicles else 0
        done = reward == -1 or self.pedestrian_pos[0] == 0 # Episode ends if hit or crossed

        if self.pedestrian_pos[0] == 0 and reward == 0:
            reward = 10  # successful crossing

        return self.get_state(), [reward], done, {}


    def get_state(self):
        # Returns the current environment state as a flat list
        # Combines pedestrian position, vehicle positions, and traffic lights

        return self.pedestrian_pos + sum(self.vehicles, []) + self.traffic_lights

    def render_string(self):
        # Creates a string representation of the grid for UI or printout
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]  # Empty grid
        for r, c in self.vehicles:
            grid[r][c] = "🚗"  # car emoji
        pr, pc = self.pedestrian_pos
        grid[pr][pc] = "🚶"  # pedestrian emoji
        output = "\n".join(" ".join(row) for row in grid)
        light_str = f"Traffic Lights (0=Red, 1=Green): {self.traffic_lights}"
        return output + "\n" + light_str

