# generate_positions.py
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Import the obstacle generation function from quadrotor.py
from quadrotor import QuadRotor12D

# Constants
KEY = 0
WORKSPACE_MIN = jnp.array([-2.8, -2.8, 0.1])
WORKSPACE_MAX = jnp.array([2.8, 2.8, 2.8])
SAFETY_MARGIN = 0.1  # Additional safety margin from obstacles

def check_collision_with_obstacles(position, obstacles, safety_margin=SAFETY_MARGIN):
    """
    Check if a position collides with any cylindrical obstacle.
    
    Args:
        position: JAX array of shape (3,) representing [x, y, z]
        obstacles: JAX array of shape (n, 5) where each row is [x, y, z_base, radius, height]
        safety_margin: Additional margin to keep from obstacles
    
    Returns:
        Boolean indicating whether there is a collision
    """
    if obstacles.shape[0] == 0:
        return False
    
    # Extract obstacle centers (x,y) and radii
    centers = obstacles[:, :2]
    radii = obstacles[:, 3]
    
    # Calculate horizontal distances to all obstacles
    deltas = centers - position[:2]
    dists = jnp.linalg.norm(deltas, axis=1)
    
    # Check if any distance is less than radius + safety margin
    return jnp.any(dists <= radii + safety_margin)

def generate_random_position(key, workspace_min, workspace_max):
    """
    Generate a random position within the workspace bounds.
    
    Args:
        key: JAX random key
        workspace_min: Minimum coordinates of workspace
        workspace_max: Maximum coordinates of workspace
        
    Returns:
        JAX array of shape (3,) representing [x, y, z]
    """
    key, subkey = jax.random.split(key)
    return jax.random.uniform(
        subkey,
        shape=(3,),
        minval=workspace_min,
        maxval=workspace_max
    )

def generate_position_pairs(key, num_pairs, obstacles, workspace_min, workspace_max, safety_margin=SAFETY_MARGIN):
    """
    Generate pairs of initial and goal positions that are collision-free.
    
    Args:
        key: JAX random key
        num_pairs: Number of position pairs to generate
        obstacles: JAX array of obstacles
        workspace_min: Minimum coordinates of workspace
        workspace_max: Maximum coordinates of workspace
        safety_margin: Additional margin to keep from obstacles
        
    Returns:
        Tuple of (initial_positions, goal_positions) where each is a JAX array
        of shape (num_pairs, 3)
    """
    initial_positions = []
    goal_positions = []
    
    for i in range(num_pairs):
        # Generate a new key for this pair
        key, subkey = jax.random.split(key)
        init_key, goal_key = jax.random.split(subkey)
        
        # Generate initial position
        collision = True
        attempts = 0
        while collision and attempts < 100:
            init_key, subkey = jax.random.split(init_key)
            init_pos = generate_random_position(subkey, workspace_min, workspace_max)
            collision = check_collision_with_obstacles(init_pos, obstacles, safety_margin)
            attempts += 1
        
        if attempts >= 100:
            print(f"Warning: Could not find collision-free initial position for pair {i}")
            continue
        
        # Generate goal position
        collision = True
        attempts = 0
        while collision and attempts < 100:
            goal_key, subkey = jax.random.split(goal_key)
            goal_pos = generate_random_position(subkey, workspace_min, workspace_max)
            collision = check_collision_with_obstacles(goal_pos, obstacles, safety_margin)
            attempts += 1
        
        if attempts >= 100:
            print(f"Warning: Could not find collision-free goal position for pair {i}")
            continue
        
        # Add to lists
        initial_positions.append(init_pos)
        goal_positions.append(goal_pos)
    
    # Convert lists to JAX arrays
    return jnp.array(initial_positions), jnp.array(goal_positions)

def visualize_positions(obstacles, initial_positions, goal_positions):
    """
    Visualize the obstacles, initial positions, and goal positions.
    
    Args:
        obstacles: JAX array of obstacles
        initial_positions: JAX array of initial positions
        goal_positions: JAX array of goal positions
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot obstacles
    if obstacles.shape[0] > 0:
        for i in range(obstacles.shape[0]):
            x, y, z_base = obstacles[i, :3]
            radius = obstacles[i, 3]
            height = obstacles[i, 4]
            
            theta = np.linspace(0, 2*np.pi, 20)
            circle_x = radius * np.cos(theta)
            circle_y = radius * np.sin(theta)
            
            # Plot bottom and top circles
            ax.plot(x + circle_x, y + circle_y, np.full_like(theta, z_base), color='r', alpha=0.6)
            ax.plot(x + circle_x, y + circle_y, np.full_like(theta, z_base + height), color='r', alpha=0.6)
            
            # Plot vertical lines
            for j in range(0, len(theta), 4):
                ax.plot([x + circle_x[j], x + circle_x[j]], 
                        [y + circle_y[j], y + circle_y[j]], 
                        [z_base, z_base + height], color='r', alpha=0.6)
    
    # Plot initial positions
    ax.scatter(initial_positions[:, 0], initial_positions[:, 1], initial_positions[:, 2], 
               color='green', marker='^', s=50, label='Initial Positions')
    
    # Plot goal positions
    ax.scatter(goal_positions[:, 0], goal_positions[:, 1], goal_positions[:, 2], 
               color='blue', marker='o', s=50, label='Goal Positions')
    
    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(WORKSPACE_MIN[0], WORKSPACE_MAX[0])
    ax.set_ylim(WORKSPACE_MIN[1], WORKSPACE_MAX[1])
    ax.set_zlim(WORKSPACE_MIN[2], WORKSPACE_MAX[2])
    ax.set_title(f'Generated {len(initial_positions)} Initial and Goal Position Pairs')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('position_pairs_visualization.png', dpi=300)
    plt.show()

def main():
    # Create a JAX random key
    key = jax.random.PRNGKey(KEY)
    quad_class = QuadRotor12D(x0 = jnp.zeros(12),xg = jnp.zeros(12)) #dummy class
    
    # Generate obstacles
    obstacles = quad_class.obstacles
    
    # Generate 200 position pairs
    key, subkey = jax.random.split(key)
    initial_positions, goal_positions = generate_position_pairs(
        subkey,
        num_pairs=200,
        obstacles=obstacles,
        workspace_min=WORKSPACE_MIN,
        workspace_max=WORKSPACE_MAX,
        safety_margin=SAFETY_MARGIN
    )
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save the positions to files
    np.save('results/initial_positions.npy', np.array(initial_positions))
    np.save('results/goal_positions.npy', np.array(goal_positions))
    
    print(f"Generated {len(initial_positions)} initial and goal position pairs")
    print(f"Saved to 'results/initial_positions.npy' and 'results/goal_positions.npy'")
    
    # Visualize a subset of the positions (first 50 for clarity)
    visualize_positions(obstacles, initial_positions[:50], goal_positions[:50])

if __name__ == "__main__":
    main()