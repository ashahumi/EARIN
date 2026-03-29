from typing import Any


import heapq
import math
import time

def manhattan_heuristic(a, b):
    """Calculates the absolute distance (Manhattan distance) between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean_heuristic(a, b):
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def greedy(maze, start, finish, heuristic_type="manhattan"):
    """
    Greedy best-first search

    Parameters:
    - maze: The 2D matrix that represents the maze with 0 represents empty space and 1 represents a wall
    - start: A tuple with the coordinates of starting position
    - finish: A tuple with the coordinates of finishing position

    - heuristic_type: Heuristic to use ("manhattan" or "euclidean")

    Returns:
    - Number of steps from start to finish, equals -1 if the path is not found
    - Viz - everything required for step-by-step vizualization
    """
    heuristics = {
        "manhattan": manhattan_heuristic,
        "euclidean": euclidean_heuristic
    }

    if heuristic_type not in heuristics:
        raise ValueError("Invalid heuristic_type. Use 'manhattan' or 'euclidean'.")

    heuristic = heuristics[heuristic_type]

    rows = len(maze)
    cols = len(maze[0])
    
    # Priority Queue stores tuples: (heuristic_cost, tie_breaker, current_node, path_taken)
    # The tie_breaker ensures Python doesn't crash trying to compare coordinates if costs are equal
    counter = 0 
    pq = []
    heapq.heappush(pq, (heuristic(start, finish), counter, start, [start]))
    
    visited = set[Any]()
    visited.add(start)
    explored_sequence = [] # Keeps track of the exact order nodes were visited
    
    # Possible movements: Right, Down, Left, Up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while pq:
        # Pop the node with the lowest heuristic cost
        h_cost, _, current, path = heapq.heappop(pq)
        explored_sequence.append(current)
        
        # Check if we reached the finish
        if current == finish:
            viz = {
                "maze": maze, 
                "start": start, 
                "finish": finish, 
                "explored": explored_sequence, 
                "path": path
            }
            # Number of steps is the length of the path minus the starting node
            return len(path) - 1, viz
            
        # Explore valid neighbors
        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)
            
            # Check boundaries and ensure it's not a wall or already visited
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0 and neighbor not in visited:
                visited.add(neighbor)
                counter += 1
                
                # Create a new path for this neighbor
                new_path = list(path)
                new_path.append(neighbor)
                
                # Add to priority queue
                heapq.heappush(pq, (heuristic(neighbor, finish), counter, neighbor, new_path))
                
    # If the queue empties and no path is found
    viz = {
        "maze": maze, 
        "start": start, 
        "finish": finish, 
        "explored": explored_sequence, 
        "path": []
    }
    return -1, viz

def vizualize(viz):
    """
    Vizualization function. Shows step by step the work of the search algorithm.
    """
    maze = viz["maze"]
    start = viz["start"]
    finish = viz["finish"]
    explored = viz["explored"]
    path = viz["path"]
    
    rows = len(maze)
    cols = len(maze[0])
    current_explored = set()
    
    print("\n--- Starting Search Visualization ---\n")
    
    for step, node in enumerate(explored):
        current_explored.add(node)
        print(f"Step {step + 1}: Visiting {node}")
        
        for r in range(rows):
            row_str = ""
            for c in range(cols):
                pos = (r, c)
                if pos == start:
                    row_str += "S "
                elif pos == finish:
                    row_str += "F "
                elif pos == node:
                    row_str += "X " # Current node being evaluated
                elif pos in current_explored:
                    row_str += ". " # Previously explored
                elif maze[r][c] == 1:
                    row_str += "+ " # Wall
                else:
                    row_str += "  " # Unexplored space
            print(row_str)
            
        print("-" * (cols * 2))
        time.sleep(0.5) # Pause for a moment to create an animation effect
        
    if path:
        print("\n--- Final Solution Path ---\n")
        for r in range(rows):
            row_str = ""
            for c in range(cols):
                pos = (r, c)
                if pos == start:
                    row_str += "S "
                elif pos == finish:
                    row_str += "F "
                elif pos in path:
                    row_str += "* " # The actual path found
                elif maze[r][c] == 1:
                    row_str += "+ "
                else:
                    row_str += "  "
            print(row_str)
    else:
        print("\nNo valid path was found.")


start_position = (0, 0)
finish_position = (4, 4)


# Example usage:
# maze 1 - path found
maze1 = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0]
]

# maze 2 - no path found
maze2 = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [0, 0, 0, 1, 0]
]

# maze 3 - "S" maze
maze3 = [
    [0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0]
]

# maze 4 - "C-ish" maze
maze4 = [
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

# maze 3 - "S" maze with blocked corner
maze5 = [
    [0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0]
]

maze6 = [
    [0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0]
]

def scenario(maze, start_position, finish_position, heuristic_type="manhattan"):
    num_steps, viz = greedy(maze, start_position, finish_position, heuristic_type)
    vizualize(viz)
    if num_steps != -1:
        print(
            f"\nPath from {start_position} to {finish_position} using greedy best-first search "
            f"with {heuristic_type} heuristic is {num_steps} steps."
        )
    else:
        print(
            f"\nNo path from {start_position} to {finish_position} exists "
            f"with {heuristic_type} heuristic."
        )

# TC: Path found
#scenario(maze1, start_position, finish_position)
#scenario(maze1, start_position, finish_position, "euclidean")

# TC: No path found
#scenario(maze2, start_position, finish_position)
#scenario(maze2, start_position, finish_position, "euclidean")

# TC: "S" maze with valid path
#scenario(maze3, (4, 0), (0, 4))
#scenario(maze3, (4, 0), (0, 4), "euclidean")

# TC 4: C-style maze with a wall between start and finish
#scenario(maze4, (0, 0), (0, 4))
#scenario(maze4, (0, 0), (0, 4), "euclidean")

# TC: Start and finish are the same
#scenario(maze4, (0, 0), (0, 0))

# TC 6: "S" maze with blocked path in corner, 2 paths available
#scenario(maze5, (0, 0), (0, 4))
#scenario(maze5, (0, 0), (0, 4), "euclidean")

# TC 7: 5 x 6 maze
scenario(maze6, (0, 0), (0, 4), "euclidean")