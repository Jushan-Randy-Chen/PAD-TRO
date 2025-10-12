# state_diffusion_chronological_projection.py

import functools
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, random
import numpy as np
import matplotlib.pyplot as plt
import jax.scipy.linalg as linalg

from tqdm import tqdm
import datetime
import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from quadrotor import QuadrotorData, QuadRotor12D, State
import util
from util import chronological_projection

print("Default backend:", jax.default_backend())

# ----------------------------- Problem data -----------------------------
data = QuadrotorData(x0=jnp.array([-1.,-1.,0.5, 0,0,0, 0,0,0, 0,0,0]),xg = jnp.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))


# ------------------------ State Diffusion with Chronological Projection ----------------------------
def state_diffusion_sampler(x0=None, 
                            xg=None, 
                            sigma_min=0.0021, 
                            sigma_max=0.2, 
                            action_samples = 128, 
                            collision_avoidance_method='cost', 
                            partial_project = True, 
                            project_last_only = False,
                            override_sigmas = False):

    rng = jax.random.PRNGKey(seed=0)
    
    # Diffusion hyperparameters
    beta0 = 1e-4
    betaT = 1e-2
    # betaT = 5e-3
    temp = 0.1
    Nsample = 256  # Number of samples per diffusion step
    Hsample = 50  # Horizon length
    # Ndiffuse = 100  # Number of diffusion steps
    Ndiffuse = 200
    
    # Use provided x0 and xg if available, otherwise use data defaults
    x0_val = x0 if x0 is not None else data.x0
    xg_val = xg if xg is not None else data.xg
    
    env = QuadRotor12D(x0_val, xg_val)
    Nx = env.observation_size
    Nu = env.action_size
    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    
    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env_jit(rng_reset)

    # Setup diffusion parameters
    betas = jnp.linspace(beta0, betaT, Ndiffuse)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas_base = jnp.sqrt(1 - alphas_bar)  # Base noise schedule over diffusion steps


    #overide sigma_max and sigma_min:
    if override_sigmas:
        sigma_max = sigmas_base[-1]
        sigma_min = sigmas_base[0]
        
    # Create 2D noise schedule: (Ndiffuse, Hsample)
    trajectory_decay_factor = 0.8  
    trajectory_weights = jnp.power(trajectory_decay_factor, jnp.arange(Hsample))
    # trajectory_weights = trajectory_decay_factor**jnp.arange(Hsample)[::-1]
    
    # Broadcast to create 2D sigmas: (Ndiffuse, Hsample)
    sigmas = sigmas_base[:, None] * trajectory_weights[None, :]
    
    # Keep initial and final states with minimal noise variation
    sigmas = sigmas.at[:, 0].set(sigmas_base * 0.1)  # Very low noise for initial state
    sigmas = sigmas.at[:, -1].set(sigmas_base * 0.1)  # Very low noise for goal state
    
    print(f"init sigma (base) = {sigmas_base[-1]:.2e}")
    print(f"final sigma (base) = {sigmas_base[0]:.2e}")
    print(f"sigma shape = {sigmas.shape}")
    
    # Initialize array (will be filled with noise)
    XN = jnp.zeros([Hsample, Nx])
    XN = XN.at[0].set(state_init.pipeline_state)  # Set initial state
    
    # Set goal state at the end of the trajectory
    XN = XN.at[-1].set(env.xg)

    @jit
    def reverse_once(carry, unused):
        """
        Single step of the reverse diffusion process with chronological projection.
        
        Args:
            carry: Tuple containing (step index, random key, current state trajectory)
            unused: Placeholder for scan function
            
        Returns:
            Updated carry tuple and mean reward
        """
        i, rng, Xbar_i = carry
        Xi = Xbar_i * jnp.sqrt(alphas_bar[i])
        
        # Generate noise samples
        rng, X0s_rng = jax.random.split(rng)
        eps_x = jax.random.normal(X0s_rng, (Nsample, Hsample, Nx)) #gaussian noise
        
        # Gaussian scaled noise with state-level variation (preserve initial and goal states)
        state_range = data.X_ub - data.X_lb
        # Use 2D sigmas: sigmas[i] has shape (Hsample,)
        scaled_noise = eps_x * (sigmas[i][:, None] * state_range[None, :])

        scaled_noise = scaled_noise.at[:, 0, :].set(0.0)
        scaled_noise = scaled_noise.at[:, -1, :].set(0.0)
        

        X0s = jnp.clip(Xbar_i + scaled_noise, data.X_lb, data.X_ub)
    

        def project_traj(traj):

            curr_sigma = jnp.mean(sigmas[i])
            
            proj_prob = jnp.clip((sigma_max - curr_sigma) / (sigma_max - sigma_min), 0.0, 1.0)
            # proj_prob_schedule = proj_prob * trajectory_weights_proj

            rng_seed = jax.random.PRNGKey(jnp.sum(traj).astype(jnp.int32))
            rand_values = jax.random.uniform(rng_seed, shape=(traj.shape[0],))
            
 
            proj_mask = jnp.where(rand_values < proj_prob, jnp.ones_like(rand_values), jnp.zeros_like(rand_values))
            proj_mask = proj_mask.at[0].set(0.0)  # Never modify initial state
            proj_mask = proj_mask.at[-1].set(0.0)  # Never modify goal state
            
  
            projected_traj = chronological_projection(traj, 
                                                        step_env_jit, 
                                                        data, 
                                                        action_samples=action_samples, 
                                                        proj_mask=proj_mask)
            
            return projected_traj

        def project_traj_always(traj):
            proj_mask = None
            projected_traj = chronological_projection(traj, 
                                                        step_env_jit, 
                                                        data, 
                                                        action_samples=action_samples, 
                                                        proj_mask=proj_mask)
            
            return projected_traj

        mean_sigma = jnp.mean(sigmas[i, :])
        X0s_projected = jax.lax.cond(
            mean_sigma < sigma_min,
            lambda: jax.vmap(project_traj_always)(X0s),  # Always project when sigma < sigma_min
            lambda: jax.lax.cond(
                mean_sigma < sigma_max,
                lambda: jax.vmap(project_traj)(X0s),  # Probabilistic projection when sigma_min <= sigma < sigma_max
                lambda: X0s  # No projection when sigma >= sigma_max
            )
        )
        
        # Compute rewards for each trajectory
        
        def compute_reward(traj):
            # Compute distance to goal
            # goal_dist = jnp.linalg.norm(traj[-1, :3] - env.xg[:3])
            traj_reward = jnp.sum(jnp.sum(jnp.square(traj[:,:3] - env.xg[:3]), axis=1))

            def compute_obstacle_penalty():

                upper_margins = data.X_ub[None, :] - traj  # Shape: [time_steps, dims]
                lower_margins = traj - data.X_lb[None, :]  # Shape: [time_steps, dims]
                
                # Apply log barrier penalty (increases as state approaches boundaries)
                # Sum across all dimensions and time steps
                boundary_penalty = -0.1 * jnp.sum(jnp.log(jnp.maximum(upper_margins, 1e-4)))
                boundary_penalty -= 0.1 * jnp.sum(jnp.log(jnp.maximum(lower_margins, 1e-4)))
                
                # Fully vectorized obstacle avoidance calculation across all timesteps and obstacles
                # Extract trajectory positions (x,y coordinates only)
                traj_positions = traj[:, :2]  # Shape: [time_steps, 2]
                
                traj_positions_reshaped = traj_positions[:, None, :]  # Shape: [time_steps, 1, 2]
                obs_positions = env.obstacles[:, :2]                   # Shape: [num_obstacles, 2]
                obs_positions_reshaped = obs_positions[None, :, :]     # Shape: [1, num_obstacles, 2]
                obs_radii = env.obstacles[:, 3]                        # Shape: [num_obstacles]
                
                # Calculate squared differences for all timesteps and all obstacles at once
                squared_diffs = jnp.square(traj_positions_reshaped - obs_positions_reshaped)

                squared_distances = jnp.sum(squared_diffs, axis=2)
                
                # Calculate center distances
                center_distances = jnp.sqrt(squared_distances)
                
        
                surface_distances = center_distances - obs_radii[None, :]

                penalties = jnp.exp(-5 * jnp.square(surface_distances))
                

                obstacle_penalty = jnp.sum(penalties)
                


                return boundary_penalty + 5*obstacle_penalty 
            
            # Use jax.lax.cond to conditionally compute obstacle penalty
            obstacle_penalty = jax.lax.cond(
                (collision_avoidance_method == 'cost') & (env.obstacles.shape[0] > 0),
                compute_obstacle_penalty,  # True branch: compute penalty
                lambda: 0.0                # False branch: return zero penalty
            )
                
            reward = -traj_reward - obstacle_penalty 
            return reward
        
        rewards = jax.vmap(compute_reward)(X0s_projected)
        # print(f'rewards has shape {rewards.shape}') #shape: (Nsample, )
        # rewards = jax.vmap(compute_reward)(X0s)
        
        # Compute weights using softmax
        logp0 = rewards / temp
        weights = jax.nn.softmax(logp0)
        # Compute weighted average of projected trajectories
        Xbar = jnp.einsum("n,nij->ij", weights, X0s_projected)
        
        # Compute score and update state trajectory
        score = 1 / (1.0 - alphas_bar[i]) * (-Xi + jnp.sqrt(alphas_bar[i]) * Xbar)
        Xim1 = 1 / jnp.sqrt(alphas[i]) * (Xi + (1.0 - alphas_bar[i]) * score)
        Xbar_im1 = Xim1 / jnp.sqrt(alphas_bar[i - 1])
        
        # Ensure initial and goal states are preserved
        Xbar_im1 = Xbar_im1.at[0].set(state_init.pipeline_state)
        Xbar_im1 = Xbar_im1.at[-1].set(env.xg)
        
        # Apply chronological projection with curriculum to ensure dynamic feasibility
        # curr_sigma = sigmas[i]  # Use the next step's sigma value
        curr_sigma = jnp.mean(sigmas[i])
        proj_prob = jnp.clip((sigma_max - curr_sigma) / (sigma_max - sigma_min), 0.0, 1.0)
        # proj_prob_schedule = proj_prob * trajectory_weights_proj
        
        # Generate random values for each timestep to determine whether to project
        # This implements partial projection where some states in the trajectory may be skipped
        rng_seed = jax.random.PRNGKey(jnp.sum(Xbar_im1).astype(jnp.int32))
        # rng_seed = jax.random.PRNGKey((jnp.sum(Xbar_im1) + i*1000).astype(jnp.int32))
        rand_values = jax.random.uniform(rng_seed, shape=(Xbar_im1.shape[0],))
        
        proj_mask = (rand_values < proj_prob).astype(jnp.float32)
        # proj_mask = (rand_values < proj_prob_schedule[i]).astype(jnp.float32)
        proj_mask = proj_mask.at[0].set(0.0)  # Never modify initial state
        proj_mask = proj_mask.at[-1].set(0.0)  # Never modify goal state
        

        if project_last_only: #with this approach, we only apply a full projection to the predicted state at the very last denoising step i=0
            return (i - 1, rng, Xbar_im1), rewards.mean()

        if partial_project:
            def apply_projection(sigma):
                # Define projection strategy based on noise level
                return jax.lax.cond(
                    sigma < sigma_min,
                    lambda: project_traj_always(Xbar_im1),  # Always project when sigma < sigma_min
                    lambda: jax.lax.cond(
                        sigma < sigma_max,
                        lambda: project_traj(Xbar_im1),  # Probabilistic projection when sigma_min <= sigma < sigma_max
                        lambda: Xbar_im1  # No projection when sigma >= sigma_max
                    )
                )
            
            # Apply the projection strategy
            Xbar_im1 = apply_projection(curr_sigma)
        else:
            # Simplified full projection logic
            def apply_full_projection(sigma):
                return jax.lax.cond(
                    sigma < sigma_max,
                    lambda: chronological_projection(Xbar_im1, step_env_jit, data, action_samples=action_samples, proj_mask=None),
                    lambda: Xbar_im1
                )
            
            # Apply the full projection strategy
            Xbar_im1 = apply_full_projection(curr_sigma)
        
        return (i - 1, rng, Xbar_im1), rewards.mean()
    
    def reverse(XN, rng):
        """
        Full reverse diffusion process.
        
        Args:
            XN: Initial noisy state trajectory
            rng: Random key
            
        Returns:
            Sequence of denoised state trajectories
        """
        Xi = XN #at the beginning Xi = XN = zeros
        Xbars = []
        
        with tqdm(range(Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                carry_once = (i, rng, Xi)
                (i, rng, Xi), rew = reverse_once(carry_once, None)
                # if i == 1 and project_last_only:
                #     Xi = chronological_projection(Xi, step_env_jit, data, proj_mask=None)
                pbar.set_postfix({"rew": f"{rew:.2e}"})
                Xbars.append(Xi)

        return jnp.array(Xbars)

    # Run the diffusion process
    rng_exp, rng = jax.random.split(rng)
    Xi_all = reverse(XN, rng_exp)
    
    # Get final trajectory
    final_traj = Xi_all[-1]
    
    # Filter states that are not changing position significantly
    position_changes = jnp.concatenate([jnp.ones((1,), dtype=bool), 
                                      jnp.linalg.norm(final_traj[1:, :3] - final_traj[:-1, :3], axis=1) > 1e-6])
    final_traj_filtered = final_traj[position_changes]
    print(f"Filtered {final_traj.shape[0] - final_traj_filtered.shape[0]} duplicate states out of {final_traj.shape[0]} total states")
    
    # Check for NaNs
    if jnp.isnan(final_traj_filtered).any():
        print("NaN detected in final trajectory... skipping")
    
    # Compute distance to goal
    distance_to_goal = jnp.linalg.norm(final_traj[-1, :3] - env.xg[:3])
    print(f"Distance to goal is {distance_to_goal}")
    
    
    return final_traj_filtered, Xi_all


# ---------------------------- Main ---------------------------------------

if __name__ == '__main__':
    # Generate date string for filenames
    date_str = datetime.datetime.now().strftime("%Y_%m_%d")
    
    # Run the state diffusion sampler with projection curriculum
    print("Running state diffusion with chronological projection curriculum...")

    sigma_min = 0.1 # Always project when noise is below this threshold
    sigma_max = 0.3     # Never project when noise is above this threshold
    # sigma_min = 0.01609512
    # sigma_max = 0.0663
    
    #TODO: this sigma_min and sigma_max won't work for all initial and final positions!!
    # Test both collision avoidance methods
    methods = ['cost']
    results = {}
    
    for method in methods:
        print(f"\n=== Testing {method.upper()} collision avoidance method ===")
        print(f"Using projection curriculum with sigma_min={sigma_min}, sigma_max={sigma_max}")
  
        final_traj, all_trajs = state_diffusion_sampler(
            x0=data.x0,
            xg=data.xg,
            sigma_min=sigma_min, 
            sigma_max=sigma_max, 
            collision_avoidance_method=method,
            action_samples=250
            
        )
        
        # Apply spline smoothing to the trajectory
        print(f"Smoothing trajectory with quintic splines...")
        smoothed_traj, traj_derivatives, _ = smooth_path_timeparam_xyz(
            final_traj
        )
        
        print(f"Original trajectory shape: {final_traj.shape}")
        print(f"Smoothed trajectory shape: {smoothed_traj.shape}")

        final_err, mean_err, trj_cl_ours = closed_loop_geometric_control(smoothed_traj, data.xg[:3], traj_derivatives)
        print(f'mean closed loop position error for smoothed trajectory is {mean_err}')
        
        # Store both original and smoothed trajectories
        results[method] = {
            'final_traj': final_traj,  # Original discrete trajectory
            'smoothed_traj': smoothed_traj,  # Spline-smoothed trajectory
            'all_trajs': all_trajs,
            'derivatives': traj_derivatives,
        }
        
        # Create environment for visualization
        env = QuadRotor12D(x0=data.x0, xg=data.xg)

        # Plot the final trajectory in 3D
        fig_front = plt.figure(figsize=(10, 8))
        ax_front = fig_front.add_subplot(111, projection='3d')
        
        # Plot both original and smoothed trajectories
        ax_front.plot(final_traj[:, 0], final_traj[:, 1], final_traj[:, 2], c='darkorange', label=f'Open Loop Trajectory', linewidth=2)
     
            #    label=f'Closed Loop Trajectory')
        ax_front.scatter(data.x0[0], data.x0[1], data.x0[2], color='green', marker='^', s=100, label='Start Position')
        ax_front.scatter(data.goal[0], data.goal[1], data.goal[2], color='yellow', marker='*', s=150, label='Goal Position')

        env.plot_scenario(ax_front)
        ax_front.set_xlabel('X(m)')
        ax_front.set_ylabel('Y(m)')
        ax_front.set_zlabel('Z(m)')
        # ax_front.set_title(f'State Diffusion Trajectory - {method.capitalize()} Collision Avoidance (Front View)')
        ax_front.legend(loc='upper right',fontsize=14)
        plt.savefig(f'results/{date_str}_state_diffusion_{method}_trajectory_front.png',dpi=300)
        
        # Create bird's eye view
        fig_bird = plt.figure(figsize=(10, 8))
        ax_bird = fig_bird.add_subplot(111, projection='3d')
        
        # Plot both original and smoothed trajectories in bird's eye view
        ax_bird.plot(final_traj[:, 0], final_traj[:, 1], final_traj[:, 2], c='darkorange',label=f'Open Loop Trajectory', linewidth=2)
      
        ax_bird.scatter(data.x0[0], data.x0[1], data.x0[2], color='green', marker='^', s=100, label='Start Position')
        ax_bird.scatter(data.goal[0], data.goal[1], data.goal[2], color='yellow', marker='*', s=150, label='Goal Position')
        # ax_bird.scatter(final_traj[-1, 0], final_traj[-1, 1], final_traj[-1, 2],  c='y', marker='x', label='Final Position')
        env.plot_scenario(ax_bird)
        ax_bird.set_xlabel('X(m)')
        ax_bird.set_ylabel('Y(m)')
        ax_bird.set_zlabel('Z(m)')
        
        
        # Set bird's eye view (top-down)
        ax_bird.view_init(elev=90, azim=-90)
        
        # Remove Z-axis ticks for aesthetics
        ax_bird.set_zticks([])
        
        # ax_bird.set_title(f'State Diffusion Trajectory - {method.capitalize()} Collision Avoidance (Bird\'s Eye View)')
        plt.savefig(f'results/{date_str}_state_diffusion_{method}_trajectory_bird.png', dpi=300)
        
        plt.show()
        

       