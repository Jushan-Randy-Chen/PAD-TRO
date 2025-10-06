# util.py
import jax
from jax import numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from quadrotor import QuadrotorData
import jax.random as random
# from scipy.interpolate import splprep, splev, interp1d
from scipy.interpolate import UnivariateSpline

data = QuadrotorData(x0=jnp.array([-1.,-1.,0.5, 0,0,0, 0,0,0, 0,0,0]),
                     xg=jnp.array([1.,1.,1., 0,0,0, 0,0,0, 0,0,0]))


def smooth_path_timeparam_xyz(traj, dt=0.1, s_factor=0.2, k=4, upsample=1, dt_out=None, eps=1e-6):
    """
    Time-parameterized smoothing with arc-length re-timing.
    - Replaces the uniform time stamps by time ∝ arc length (reduces sharp turns).
    - Fits per-axis UnivariateSplines in *time*; derivatives are d/dt directly.
    - Returns x,y,z and time derivatives up to 4th order.
    - Also returns tangent heading psi = atan2(vy, vx) and its first two time derivatives.

    Args:
        traj: (T, D) array; columns 0:3 are x,y,z. Other columns are ignored.
        dt:    nominal source step (used only to scale output horizon if dt_out not given)
        s_factor: smoothing knob (~0.05…1.0). Larger ⇒ smoother (more deviation).
        k: spline degree (3=cubic is robust). If you need snap, 5 is okay if T is large.
        upsample: integer; if dt_out is None, output step = dt/upsample.
        dt_out: explicit output step (overrides upsample if not None).
        eps: small number to avoid divide-by-zero.

    Returns:
        traj_out: (N, D) array with x,y,z in [:,0:3] and v in [:,6:9] (others zero)
        deriv: dict with x_dot, x_ddot, x_dddot, x_ddddot (and y/z analogs),
               psi, psi_dot, psi_ddot, and t.
        t_out: time grid for outputs (length N)
    """
    T = traj.shape[0]
    xyz = np.asarray(traj[:, :3])
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    psi = traj[:,3]

    # --- 1) Build arc-length-based timestamps (monotone, reduces spatial corners)
    diffs = np.diff(xyz, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)                 # shape (T-1,)
    u = np.concatenate([[0.0], np.cumsum(seglen)])         # cumulative arc length
    total_len = u[-1]
    if total_len > 0:
        u_norm = u / total_len
    else:
        u_norm = np.linspace(0.0, 1.0, T)                  # degenerate path

    # Scale to a similar total duration as original: (T-1)*dt
    t = u_norm * max((T-1)*dt, dt)                         # retimed time stamps

    # --- 2) Choose smoothing and degree
    # Heuristic: scale s to the path length; increase s_factor if corners persist
    s = (s_factor * max(total_len, 1.0))**2
    # Degree safety: need T > k
    if T <= k:
        k_eff = k-1
    else:    
        k_eff = k                       

    
    # --- 3) Fit splines in time
    sx = UnivariateSpline(t, x, s=s, k=k_eff)
    sy = UnivariateSpline(t, y, s=s, k=k_eff)
    sz = UnivariateSpline(t, z, s=s, k=k_eff)

    # --- 4) Output time grid
    if dt_out is None:
        dt_out = dt / max(1, upsample)
    t_out = np.arange(t[0], t[-1] + 0.5*dt_out, dt_out)

    # --- 5) Evaluate positions and time derivatives (no chain rule needed)
    x_s  = sx(t_out);  y_s  = sy(t_out);  z_s  = sz(t_out)
    vx   = sx.derivative(1)(t_out);  vy   = sy.derivative(1)(t_out);  vz   = sz.derivative(1)(t_out)
    ax   = sx.derivative(2)(t_out);  ay   = sy.derivative(2)(t_out);  az   = sz.derivative(2)(t_out)
    jx   = sx.derivative(3)(t_out);  jy   = sy.derivative(3)(t_out);  jz   = sz.derivative(3)(t_out)
    sx4  = sx.derivative(4)(t_out);  sy4  = sy.derivative(4)(t_out);  sz4  = sz.derivative(4)(t_out)

    # --- 6) Tangent heading psi and its time derivatives from (vx,vy,ax,ay,jx,jy)
    # psi = np.arctan2(vy, vx)
    # D   = vx*vx + vy*vy + eps
    # psi_dot = (vx*ay - vy*ax) / D
    # N   = vx*ay - vy*ax
    # dN  = vx*jy - vy*jx
    # dD  = 2.0*(vx*ax + vy*ay)
    # psi_ddot = (dN*D - N*dD) / (D*D)
    psi_raw = np.asarray(psi)                 # shape (T,)
    psi_unw = np.unwrap(psi_raw)              # continuous (adds ±2π when needed)

    # fit time spline on psi_unw (use your t, k_eff, s)
    spsi = UnivariateSpline(t, psi_unw, s=s, k=k_eff)

    psi     = spsi(t_out)
    psi_dot    = spsi.derivative(1)(t_out)
    psi_ddot   = spsi.derivative(2)(t_out)

    # --- 7) Package outputs (keep other state dims zero except vel slot)
    Dfull = traj.shape[1]
    traj_out = np.zeros((t_out.size, Dfull))
    traj_out[:, 0:3] = np.column_stack([x_s, y_s, z_s])
    traj_out[:, 3] = psi
    traj_out[:, 6:9] = np.column_stack([vx, vy, vz])

    deriv = dict(
        t=t_out,
        x_dot=vx, y_dot=vy, z_dot=vz,
        x_ddot=ax, y_ddot=ay, z_ddot=az,
        x_dddot=jx, y_dddot=jy, z_dddot=jz,
        x_ddddot=sx4, y_ddddot=sy4, z_ddddot=sz4,
        psi=psi, psi_dot=psi_dot, psi_ddot=psi_ddot
    )
    return traj_out, deriv, t_out


def rollout_us(step_env, state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, (state.reward, state.pipeline_state)
    
    _, (rews, pipline_states) = jax.lax.scan(step, state, us)
    return rews, pipline_states


def rollout_us_terminal(step_env, state, us, goal_position, terminal_weight=100.0):
    """Rollout with explicit terminal cost handling."""
    def step_with_index(carry, u_and_index):
        state, step_count = carry
        u, index = u_and_index
        state = step_env(state, u)
        
        # Check if this is the final step
        is_final = index == (us.shape[0] - 1)
        
        # Calculate terminal reward (negative cost) for final step
        terminal_cost = jnp.where(
            is_final,
            -terminal_weight * jnp.linalg.norm(state.pipeline_state[:3] - goal_position[:3])**2,
            0.0
        )
        
        # Add terminal cost to reward
        total_reward = state.reward + terminal_cost
        
        return (state, step_count + 1), (total_reward, state.pipeline_state)
    
    # Create indices for each control input
    indices = jnp.arange(us.shape[0])
    us_with_indices = (us, indices)
    
    _, (rews, pipeline_states) = jax.lax.scan(
        step_with_index, 
        (state, 0), 
        us_with_indices
    )
    
    return rews, pipeline_states



def create_diffusion_animation(all_trajectories, env, save_path=None):
    """
    Create animation showing the diffusion process with 3D and top-down views.
    
    Args:
        all_trajectories: Array of shape (n_steps, n_timesteps, state_dim)
        env: QuadRotor12D environment
        save_path: Optional path to save the animation
    """
    fig = plt.figure(figsize=(18, 8))
    
    # Create 3D subplot
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title('3D View: Trajectory Denoising')
    
    # Create 2D top-down subplot
    ax_2d = fig.add_subplot(122)
    ax_2d.set_xlabel('X (m)')
    ax_2d.set_ylabel('Y (m)')
    ax_2d.set_title('Top-Down View: Trajectory Denoising')
    ax_2d.set_aspect('equal')
    ax_2d.grid(True, alpha=0.3)
    
    # Set axis limits based on all trajectories
    all_positions = all_trajectories[:, :, :3]  # Extract x, y, z positions
    margin = 0.5
    x_min, x_max = all_positions[:, :, 0].min() - margin, all_positions[:, :, 0].max() + margin
    y_min, y_max = all_positions[:, :, 1].min() - margin, all_positions[:, :, 1].max() + margin
    z_min, z_max = all_positions[:, :, 2].min() - margin, all_positions[:, :, 2].max() + margin
    
    # Set 3D axis limits
    ax_3d.set_xlim(x_min, x_max)
    ax_3d.set_ylim(y_min, y_max)
    ax_3d.set_zlim(z_min, z_max)
    
    # Set 2D axis limits
    ax_2d.set_xlim(x_min, x_max)
    ax_2d.set_ylim(y_min, y_max)
    
    # Plot static elements on 3D view
    ax_3d.scatter(data.x0[0], data.x0[1], data.x0[2], c='g', marker='o', s=100, label='Start Position')
    ax_3d.scatter(data.goal[0], data.goal[1], data.goal[2], c='r', marker='*', s=200, label='Goal Position')
    
    # Plot static elements on 2D view
    ax_2d.scatter(data.x0[0], data.x0[1], c='g', marker='o', s=100, label='Start Position')
    ax_2d.scatter(data.goal[0], data.goal[1], c='r', marker='*', s=200, label='Goal Position')
    
    # Plot environment obstacles on both views
    env.plot_scenario(ax_3d)
    # For 2D view, we'll plot obstacles as circles (top-down view)
    # This assumes obstacles are spherical - you may need to adjust based on your environment
    try:
        if hasattr(env, 'obstacles'):
            for obs in env.obstacles:
                if hasattr(obs, 'center') and hasattr(obs, 'radius'):
                    circle = plt.Circle((obs.center[0], obs.center[1]), obs.radius, 
                                      color='red', alpha=0.3, label='Obstacle')
                    ax_2d.add_patch(circle)
    except:
        pass  # Skip if obstacle plotting fails
    
    # Initialize empty lines for trajectories
    line_3d, = ax_3d.plot([], [], [], 'b-', linewidth=2, alpha=0.8)
    line_2d, = ax_2d.plot([], [], 'b-', linewidth=2, alpha=0.8)
    
    # Initialize scatter plots for current positions
    scatter_3d = ax_3d.scatter([], [], [], c='blue', marker='o', s=50, alpha=0.6)
    scatter_2d = ax_2d.scatter([], [], c='blue', marker='o', s=50, alpha=0.6)
    
    # Text for diffusion step on 3D plot
    step_text = ax_3d.text2D(0.05, 0.95, '', transform=ax_3d.transAxes, fontsize=12, 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Text for altitude info on 2D plot
    altitude_text = ax_2d.text(0.05, 0.95, '', transform=ax_2d.transAxes, fontsize=10,
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    def animate(frame):
        if frame < len(all_trajectories):
            # Current trajectory
            traj = all_trajectories[frame]
            positions = traj[:, :3]  # Extract x, y, z positions
            
            # Update 3D trajectory line
            line_3d.set_data_3d(positions[:, 0], positions[:, 1], positions[:, 2])
            
            # Update 2D trajectory line (top-down view)
            line_2d.set_data(positions[:, 0], positions[:, 1])
            
            # Update scatter plots for current positions
            scatter_3d._offsets3d = (positions[-1:, 0], positions[-1:, 1], positions[-1:, 2])
            scatter_2d.set_offsets(jnp.column_stack([positions[-1:, 0], positions[-1:, 1]]))
            
            # Update step text
            distance_to_goal = jnp.linalg.norm(positions[-1] - data.goal[:3])
            step_text.set_text(f'Diffusion Step: {frame + 1}/{len(all_trajectories)}\n'
                             f'Distance to Goal: {distance_to_goal:.3f}m')
            
            # Update altitude text for 2D view
            current_altitude = positions[-1, 2]
            goal_altitude = data.goal[2]
            altitude_text.set_text(f'Current Altitude: {current_altitude:.3f}m\n'
                                 f'Goal Altitude: {goal_altitude:.3f}m\n'
                                 f'Alt. Error: {abs(current_altitude - goal_altitude):.3f}m')
            
            # Change color as we progress (from red/noisy to blue/clean)
            progress = frame / len(all_trajectories)
            color = plt.cm.RdYlBu(progress)
            line_3d.set_color(color)
            line_2d.set_color(color)
            
        return line_3d, line_2d, scatter_3d, scatter_2d, step_text, altitude_text
    
    # Create animation
    n_frames = len(all_trajectories)
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, 
                                 interval=200, blit=False, repeat=True)
    
    # Add legends
    ax_3d.legend(loc='upper right')
    ax_2d.legend(loc='upper right')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save animation if path provided
    if save_path:
        print(f"Saving animation to {save_path}")
        anim.save(save_path, writer='pillow', fps=5)
        print("Animation saved!")
    
    return anim


def chronological_projection(state_traj, step_env, data, action_samples=128, proj_mask=None):
    """
    Chronologically project a state trajectory to enforce dynamic feasibility.
    
    This function projects each state in the trajectory back to the first state
    by ensuring that each transition is dynamically feasible using a convex
    approximation of the action set.
    
    Args:
        state_traj: State trajectory of shape (Hsample+1, Nx)
        step_env: Environment step function
        data: Problem data containing bounds and dynamics info
        action_samples: Number of action samples for projection
        proj_mask: Optional mask indicating which timesteps to project (1.0 = project, 0.0 = don't project)
                  If None, all timesteps are projected
    
    Returns:
        Projected state trajectory that is dynamically feasible
    """
    projected_traj = state_traj.copy()
    
    # If no projection mask is provided, project all timesteps
    # Use JAX's where to handle the None case in a tracing-compatible way
    proj_mask = jnp.ones(state_traj.shape[0]) if proj_mask is None else proj_mask
    proj_mask = proj_mask.at[0].set(0)
    proj_mask = proj_mask.at[-1].set(0)
    
    # Define a function for a single projection step
    def projection_step(t, traj):
        prev_state = traj[t-1]
        target_state = state_traj[t]
        
        # Only project if the mask indicates to do so
        # Use jax.lax.cond to make this JAX-compatible
        def do_projection(_):
            return project_onto_reachable_set(
                prev_state, target_state, step_env, data, action_samples=action_samples
            )
            
        def skip_projection(_):
            return state_traj[t]
            
        # Conditionally project based on the mask
        new_state = jax.lax.cond(
            proj_mask[t] > 0.0,
            do_projection,
            skip_projection,
            operand=None
        )
        
        return traj.at[t].set(new_state)
    
    # Use JAX's fori_loop instead of Python for loop
    projected_traj = jax.lax.fori_loop(
        1, state_traj.shape[0], projection_step, projected_traj
    )
    
    return projected_traj


def project_onto_reachable_set(prev_state, target_state, step_env, data, action_samples=128):
    """
    Project a target state onto the reachable set from a previous state.
    
    Uses a convex approximation of the action set to find the closest
    dynamically feasible state to the target.
    
    Args:
        prev_state: Previous state (Nx,)
        target_state: Target state to project (Nx,)
        step_env: Environment step function
        data: Problem data
    
    Returns:
        Projected state that is reachable from prev_state
    """
    # Sample actions from convex approximation of action set
    action_samples = convex_action_approximation(data, n_samples=action_samples)
    
    # Compute reachable states
    def compute_next_state(action):
        from quadrotor import State
        state_obj = State(prev_state, prev_state, 0.0, 0.0)
        next_state_obj = step_env(state_obj, action)
        return next_state_obj.pipeline_state
    
    reachable_states = jax.vmap(compute_next_state)(action_samples)
    
    # Find closest reachable state to target
    distances = jnp.linalg.norm(reachable_states - target_state[None, :], axis=1)
    best_idx = jnp.argmin(distances)
    
    return reachable_states[best_idx]


def convex_action_approximation(data, n_samples=128):
    """
    Generate samples from a convex approximation of the action set.
    
    For the quadrotor, this creates a polytopic approximation of the
    feasible control inputs.
    
    Args:
        data: Problem data containing action bounds
        n_samples: Number of action samples to generate
    
    Returns:
        Action samples of shape (n_samples, Nu)
    """
    # For simplicity, use uniform sampling within action bounds
    # In practice, this could be a more sophisticated polytopic approximation
    key = jax.random.PRNGKey(0)  # Fixed seed for deterministic behavior
    
    # Generate random samples within action bounds
    action_samples = jax.random.uniform(
        key, 
        shape=(n_samples, len(data.U_lb)),
        minval=data.U_lb,
        maxval=data.U_ub
    )
    
    return action_samples


def project_with_predicted_u(prev_state, target_state, step_env, u_predicted):

    def compute_next_state(action):
        from quadrotor import State
        state_obj = State(prev_state, prev_state, 0.0, 0.0)
        next_state_obj = step_env(state_obj, action)
        return next_state_obj.pipeline_state
    
    reachable_state = compute_next_state(u_predicted) #tricky part: can we use a weighted average of reachable_state and our target_state?

    weight1 = 0.5
    weight2 = 0.5

    return reachable_state*weight1 + target_state*weight2



def state_to_control_mapping(state_traj, step_env, data, action_samples = 256):
    """
    Infer control sequence from a state trajectory using approximate inverse dynamics.
    
    This function attempts to find the control inputs that would produce
    the given state trajectory.
    
    Args:
        state_traj: State trajectory of shape (Hsample+1, Nx)
        step_env: Environment step function
        data: Problem data
    
    Returns:
        Control sequence of shape (Hsample, Nu)
    """
    n_steps = state_traj.shape[0] - 1
    control_seq = jnp.zeros((n_steps, len(data.U_lb)))
    
    for t in range(n_steps):
        current_state = state_traj[t]
        next_state = state_traj[t + 1]
        
        # Find control that best achieves the transition
        # optimal_control = find_optimal_control(
        #     current_state, next_state, step_env, data, action_samples=action_samples
        # )
        optimal_control = black_box_inverse_dyanmics(current_state, next_state, step_env, data, action_samples=action_samples)
        control_seq = control_seq.at[t].set(optimal_control)
    
    return control_seq


def find_optimal_control(current_state, target_next_state, step_env, data, action_samples=256):
    """
    Find the optimal control to transition from current_state to target_next_state.
    This is an approximate inverse dynamics model
    
    Args:
        current_state: Current state (Nx,)
        target_next_state: Desired next state (Nx,)
        step_env: Environment step function
        data: Problem data
    
    Returns:
        Optimal control input (Nu,)
    """
    # Use a simple grid search for now (could be replaced with optimization)
    action_candidates = convex_action_approximation(data, n_samples=action_samples)
    
    def compute_state_error(action):
        from quadrotor import State
        state_obj = State(current_state, current_state, 0.0, 0.0)
        next_state_obj = step_env(state_obj, action)
        predicted_next_state = next_state_obj.pipeline_state
        return jnp.linalg.norm(predicted_next_state - target_next_state)**2
    
    errors = jax.vmap(compute_state_error)(action_candidates)
    best_idx = jnp.argmin(errors)
    
    return action_candidates[best_idx]