# quadrotor.py
import jax
from jax import numpy as jnp
from jax._src.lax.control_flow.loops import Y
from flax import struct
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

KEY=0

def generate_obstacles_grid(key, grid_div, region_min, region_max, min_radius, max_radius, min_clearance=0.5):
    """Generate 3D cylindrical obstacles in a grid layout.
    
    Args:
        key: JAX random key
        grid_div: Number of grid divisions in x and y directions
        region_min: Minimum coordinate value for the region
        region_max: Maximum coordinate value for the region
        min_radius: Minimum radius for cylinders
        max_radius: Maximum radius for cylinders
        height_min: Minimum height for cylinders
        height_max: Maximum height for cylinders
        min_clearance: Minimum clearance between obstacles
        
    Returns:
        JAX array of obstacles with shape (n, 5) where each row contains:
        [x, y, z_base, radius, height] for each cylindrical obstacle
    """
    cell_size = (region_max - region_min) / grid_div
    
    # Pre-allocate maximum possible number of obstacles
    max_obstacles = grid_div * grid_div
    obstacles_list = []
    
    for i in range(grid_div):
        for j in range(grid_div):
            # Generate a new key for this cell
            key, subkey = jax.random.split(key)
            
            # Split keys for different random values
            radius_key, x_key, y_key, z_key, height_key = jax.random.split(subkey, 5)
            
            # Generate random radius for cylinder (height is fixed from ground to ceiling)
            radius = jax.random.uniform(radius_key, minval=min_radius, maxval=max_radius)
            height = 3.0  # Fixed height from ground to ceiling
            margin = radius + 0.2
            
            # Calculate valid range for obstacle center (x,y)
            x_min = region_min + i * cell_size + margin
            x_max = region_min + (i + 1) * cell_size - margin
            y_min = region_min + j * cell_size + margin
            y_max = region_min + (j + 1) * cell_size - margin
            
            # Generate random position (x,y) for cylinder base center
            x = jax.random.uniform(x_key, minval=x_min, maxval=x_max)
            y = jax.random.uniform(y_key, minval=y_min, maxval=y_max)
            
            # Set z-coordinate for cylinder base to ground level
            z_base = 0.0  # Fixed at ground level
            
            # Check clearance from existing obstacles
            if obstacles_list:
                # Extract existing cylinder data
                centers_xy = jnp.array([[ox, oy] for ox, oy, _, _, _ in obstacles_list])
                radii = jnp.array([r for _, _, _, r, _ in obstacles_list])
                
                # Calculate horizontal distances to all existing obstacles
                dists_xy = jnp.sqrt(jnp.sum((centers_xy - jnp.array([x, y]))**2, axis=1))
                min_dists = radii + radius + min_clearance
                
                too_close = jnp.any(dists_xy < min_dists)
                if not too_close:
                    obstacles_list.append((x, y, z_base, radius, height))
            else:
                obstacles_list.append((x, y, z_base, radius, height))
    
    if obstacles_list:
        obstacles_array = jnp.array(obstacles_list)
    else:
        obstacles_array = jnp.zeros((0, 5))
        
    return obstacles_array

class QuadrotorData:
    """
    Holds quadrotor OCP data: dynamics parameters, obstacles, bounds, horizon, etc.
    """
    def __init__(self, x0, xg):
        self.m = 1.0
        self.g = 9.81
        self.Ix, self.Iy, self.Iz = 0.01, 0.01, 0.02
        self.dt = 0.1
        self.N = 50
        
        self.X_lb = jnp.array([-3, -3, 0,   -jnp.pi, -jnp.pi/4, -jnp.pi/4,  -2, -2, -2,  -1, -1, -1])
        self.X_ub = jnp.array([ 3,  3, 3,    jnp.pi,  jnp.pi/4,  jnp.pi/4,   2,  2,  2,   1,  1,  1])
        self.U_lb = jnp.array([-2, -2, -2,   0])
        self.U_ub = jnp.array([ 2,  2,  2,  20])
        
        # self.x0   = jnp.array([-1.,-1.,0.5, 0,0,0, 0,0,0, 0,0,0])
        self.x0 = x0
        self.xg = xg
        self.goal = self.xg[:3]
        # self.xg = jnp.array([1.,1.,1., 0,0,0, 0,0,0, 0,0,0])

data = QuadrotorData(x0=jnp.array([-1.,-1.,0.5, 0,0,0, 0,0,0, 0,0,0]),
                     xg=jnp.array([1.,1.,1., 0,0,0, 0,0,0, 0,0,0]))

def quadrotor_dynamics(x_t, u_t):
    x_pos, y_pos, z_pos = x_t[0], x_t[1], x_t[2]
    psi, theta, phi = x_t[3], x_t[4], x_t[5]
    vx, vy, vz = x_t[6], x_t[7], x_t[8]
    p, q, r = x_t[9], x_t[10], x_t[11]
    
    tau_x, tau_y, tau_z = u_t[0], u_t[1], u_t[2]
    F = u_t[3]
    
    c_phi, s_phi = jnp.cos(phi), jnp.sin(phi)
    c_theta, s_theta = jnp.cos(theta), jnp.sin(theta)
    c_psi, s_psi = jnp.cos(psi), jnp.sin(psi)
    
    x_dot = vx
    y_dot = vy
    z_dot = vz
    
    psi_dot = (q * s_phi) / (c_theta) + (r * c_phi) / (c_theta)
    theta_dot = q * c_phi - r * s_phi
    phi_dot = p + q * s_phi * jnp.tan(theta) + r * c_phi * jnp.tan(theta)
    
    vx_dot = (F/data.m)  * (c_phi * s_theta * c_psi + s_phi * s_psi)
    vy_dot = (F/data.m) * (c_phi * s_theta * s_psi - s_phi * c_psi)
    vz_dot = -data.g + (F/data.m) * c_phi * c_theta
    
    p_dot = (data.Iy * q * r - data.Iz * q * r + tau_x) / data.Ix
    q_dot = (-data.Ix * p * r + data.Iz * p * r + tau_y) / data.Iy
    r_dot = (data.Ix * p * q - data.Iy * p * q + tau_z) / data.Iz
    
    return jnp.array([
        x_dot, y_dot, z_dot,
        psi_dot, theta_dot, phi_dot,
        vx_dot, vy_dot, vz_dot,
        p_dot, q_dot, r_dot
    ])

def car_dynamics(x, u):
    return jnp.array(
        [
            u[1] * jnp.sin(x[2])*3.0,
            u[1] * jnp.cos(x[2])*3.0,
            u[0] * jnp.pi / 3 * 2.0,
        ]
    )

def rk4(dynamics, x, u, dt):
    k1 = dynamics(x, u)
    k2 = dynamics(x + dt / 2 * k1, u)
    k3 = dynamics(x + dt / 2 * k2, u)
    k4 = dynamics(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

SAFETY_MARGIN = 0
def check_collision(x, obstacles):
    centers = obstacles[:, :2]
    radii   = obstacles[:, 3]
    deltas = centers - x[:2] 
    dists  = jnp.linalg.norm(deltas, axis=1)
    return jnp.any(dists <= radii + SAFETY_MARGIN)

@struct.dataclass
class State:
    pipeline_state: jnp.ndarray
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray

class QuadRotor12D:
    def __init__(self,x0, xg):
        self.dt = 0.1
        self.H = 50
        key = jax.random.PRNGKey(KEY)
        self.obstacles = generate_obstacles_grid(key,
                                                grid_div=4, 
                                                region_min=-2.0, 
                                                region_max=2.0, 
                                                min_radius=0.1, 
                                                max_radius=0.2, 
                                                min_clearance=0.3)
                                
        if self.obstacles.shape[0] > 0:
            self.obs_positions = self.obstacles[:, :3]
            self.obs_radii = self.obstacles[:, 3]
            self.obs_heights = self.obstacles[:, 4]
        else:
            self.obs_positions = jnp.zeros((0, 3))
            self.obs_radii = jnp.zeros((0,))
            self.obs_heights = jnp.zeros((0,))

        self.x0 = x0
        self.xg = xg

    def reset(self, rng: jax.Array):
        return State(self.x0, self.x0, 0.0, 0.0)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, data.U_lb, data.U_ub)
        q = state.pipeline_state
        q_new = rk4(quadrotor_dynamics, state.pipeline_state, action, self.dt)
        q_new = jnp.clip(q_new, data.X_lb, data.X_ub)
        
        collide = check_collision(q_new, self.obstacles)
        q = jnp.where(collide, q, q_new)
        reward = self.get_reward(q, action)

        return state.replace(pipeline_state=q, obs=q, reward=reward, done=0.0)
    
    # @partial(jax.jit, static_argnums=(0,))
    # def step_eval(self, state: State, action: jax.Array) -> State:
    #     action = jnp.clip(action, data.U_lb, data.U_ub)
    #     q = state.pipeline_state
    #     q_new = rk4(quadrotor_dynamics, state.pipeline_state, action, self.dt)
    #     q_new = jnp.clip(q_new, data.X_lb, data.X_ub)

    #     # collide = check_collision(q_new, self.obstacles)
    #     # q = jnp.where(collide, q, q_new)
    #     # reward = self.get_reward(q, action)

    #     return state.replace(pipeline_state=q_new, obs=q_new, reward=0, done=0.0)

    @partial(jax.jit, static_argnums=(0,))
    def get_reward(self, q, u):
        reward = -jnp.linalg.norm(q[:3] - self.xg[:3])**2 - 0.001*jnp.linalg.norm(u[:3])
        return reward
    
    @property
    def action_size(self):
        return 4

    @property
    def observation_size(self):
        return 12

    def _obstacle_cost(self, x: jnp.ndarray) -> jnp.ndarray:
        def scan_fn(total_cost: float, obs_pos: jnp.ndarray):
            cost = 0.0
            for obs in obs_pos:
                x,y,z_base,radius,height = obs
                obs_pos = jnp.array([x,y])
                cost += jnp.exp(-5 * jnp.linalg.norm(x[0:2] - obs_pos - radius) ** 2)
            return total_cost + cost, None

        total_cost, _ = jax.lax.scan(scan_fn, 0.0, self.obstacles)
        return total_cost
        
    def plot_scenario(self, ax=None) -> None:
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        if self.obstacles.shape[0] > 0:
            for i in range(self.obstacles.shape[0]):
                x, y, z_base = self.obs_positions[i]
                radius = self.obs_radii[i]
                height = self.obs_heights[i]
                
                theta = jnp.linspace(0, 2*jnp.pi, 20)
                circle_x = radius * jnp.cos(theta)
                circle_y = radius * jnp.sin(theta)
                
                ax.plot(x + circle_x, y + circle_y, jnp.full_like(theta, 0.0), color='r', alpha=0.6)
                ax.plot(x + circle_x, y + circle_y, jnp.full_like(theta, 3.0), color='r', alpha=0.6)
                
                for j in range(0, len(theta), 4):
                    ax.plot([x + circle_x[j], x + circle_x[j]], 
                            [y + circle_y[j], y + circle_y[j]], 
                            [0.0, 3.0], color='r', alpha=0.6)
                
                ax.scatter(x, y, 0.0, color='r', s=20, alpha=0.8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(0, 3)

    def plot_dist_to_obs(self, trj, ax=None) -> None:
        """
        Plot ‖x(t)–c_i‖ for every obstacle i along a trajectory.

        Parameters
        ----------
        trj : (T, 12) array
            Optimised state trajectory (only x,y are used here).
        ax  : matplotlib Axes, optional
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        T   = trj.shape[0]
        ts  = np.arange(T)

        if self.obstacles.size == 0:          # nothing to draw
            ax.set_xlabel("Time index")
            ax.set_ylabel("Distance to obstacles [m]")
            return

        for k, obs in enumerate(np.array(self.obstacles)):   # convert to NumPy
            centre = obs[:2]                                 # (x_c , y_c)
            radius = float(obs[3])                           # scalar

            # vectorised centre-to-surface distance in the x-y plane
            center_to_center_dists = np.linalg.norm(trj[:, :2] - centre, axis=1)  # distance from trajectory to obstacle center
            dists = center_to_center_dists - radius  # subtract radius to get surface distance
            ax.plot(ts, dists, label=f"obs {k+1}")
        
        # ax.hlines(y=radius + SAFETY_MARGIN,
        #         xmin=0, xmax=T-1,
        #         colors="r", linestyles="--", label='Safety Theshold Distance')

        ax.set_xlabel("Time index")
        ax.set_ylabel("Distance to obstacles [m]")
        ax.set_title("Clearance to each obstacle over the trajectory")
        ax.legend()

