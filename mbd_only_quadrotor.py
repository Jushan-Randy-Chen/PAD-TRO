
"""
This is adapted from : https://github.com/LeCAR-Lab/model-based-diffusion/tree/main/mbd/planners

"""
import functools
import os
import sys
import jax
import jax.numpy as jnp
from jax import jit, grad, jacfwd, vmap, random
import numpy as np
import matplotlib.pyplot as plt
import jax.scipy.linalg as linalg
import util
from quadrotor import *
from tqdm import tqdm
from util import rollout_us_terminal, create_diffusion_animation
import datetime
import pickle
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.animation as animation

print("Default backend:", jax.default_backend())
# ----------------------------- Problem data -----------------------------
data = QuadrotorData(x0 =jnp.array([-1.,-1.,0.5, 0,0,0, 0,0,0, 0,0,0]),xg = jnp.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

# ------------------------ Phase II: Diffusion ----------------------------
def sampler(x0, xg):

    rng = jax.random.PRNGKey(seed=0)
    
    # Diffusion hyperparameters
    beta0=1e-4
    betaT=1e-2
    temp=0.1
    Nsample = 256
    Hsample = 49
    Ndiffuse = 200
    
    env = QuadRotor12D(x0, xg)
    Nx = env.observation_size
    Nu = env.action_size
    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    rollout_us = jax.jit(functools.partial(util.rollout_us, step_env_jit))
    
    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env_jit(rng_reset)
    
    betas = jnp.linspace(beta0, betaT, Ndiffuse)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1 - alphas_bar)
    print(f"init sigma = {sigmas[-1]:.2e}")

    YN = jnp.zeros([Hsample, Nu]) #this is the control trajectory only
    
    @jit
    def reverse_once(carry, unused):
        i, rng, Ybar_i = carry
        Yi = Ybar_i * jnp.sqrt(alphas_bar[i])

        rng, Y0s_rng = jax.random.split(rng)
        eps_u = jax.random.normal(Y0s_rng, (Nsample, Hsample, Nu))
        
        # Gaussian scaled noise
        control_range = data.U_ub - data.U_lb   
        scaled_noise = eps_u * (sigmas[i] * control_range)[None, None, :] # broadcast to correct shape
        Y0s = jnp.clip(Ybar_i + scaled_noise, data.U_lb, data.U_ub)
        
        # In the reverse_once function, replace the rollout_us call:
        # rews, qs = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, Y0s)
        
        # With the new rollout_us_terminal function that adds penalty on the temrinal distance to goal:
        rews, qs = jax.vmap(
            lambda state, us: rollout_us_terminal(step_env_jit, state, us, env.xg, terminal_weight=10.0),
            in_axes=(None, 0)
        )(state_init, Y0s)
        # print(rews.shape)
        rews = rews.mean(axis=-1)
        
        logp0 = rews / temp
        weights = jax.nn.softmax(logp0)
        Ybar = jnp.einsum("n,nij->ij", weights, Y0s)

        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)
        Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)
        Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i - 1]) 
        return (i - 1, rng, Ybar_im1), rews.mean()
        
    def reverse(YN, rng):
        Yi = YN
        Ybars = []
        Ybars_all = []
        
        with tqdm(range(Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                carry_once = (i, rng, Yi)
                (i, rng, Yi), rew = reverse_once(carry_once, None)
                # print(f'current denoised control input is {Yi}, has shape {Yi.shape}')
                pbar.set_postfix({"rew": f"{rew:.2e}"})
                Ybars.append(Yi)

                # Recording intermediate denoised trajectories for animation 
                xs = jnp.array([state_init.pipeline_state])
                state = state_init
                for t in range(Yi.shape[0]):
                    state = step_env_jit(state, Yi[t])
                    xs = jnp.concatenate([xs, state.pipeline_state[None]], axis=0)
                Ybars_all.append(xs)

        return jnp.array(Ybars), jnp.array(Ybars_all)

    rng_exp, rng = jax.random.split(rng)
    Yi, Yi_all = reverse(YN, rng_exp)
    print(f'Yi has shape {Yi.shape}')
    # rollout final trajectory
    print(f'roll out begins...')
    xs = jnp.array([state_init.pipeline_state])
    state = state_init
    print(f'initial state is {xs}')
    for t in range(Yi.shape[1]):
        state = step_env_jit(state, Yi[-1, t])
        xs = jnp.concatenate([xs, state.pipeline_state[None]], axis=0)
        
        # if np.linalg.norm(env.xg[:3] - xs[-1][:3]) <= 0.15:
        #     print(f'converged to goal position')
        #     break

    
    if jnp.isnan(xs).any():
        print(" nan in Phase II… skipping")

    distance_to_goal = jnp.linalg.norm(xs[-1, :3] - env.xg[:3])
    print(f"Distance to goal is {distance_to_goal}")

    return xs, Yi_all, Yi[-1,:]


# ---------------------------- Main ---------------------------------------

if __name__ == '__main__':
    # Generate date string for filenames
    date_str = datetime.datetime.now().strftime("%Y_%m_%d")
    x0=jnp.array([-1.,-1.,0.5, 0,0,0, 0,0,0, 0,0,0])
    xg=jnp.array([1.,1.,1., 0,0,0, 0,0,0, 0,0,0])

    trajs, trajs_all, control_seq = sampler(x0, xg)
    
    print(f'rolled out trajectory: {trajs[:,:3]}')
    # print(f'denoised control sequence: {control_seq[:,:]}') 
    env = QuadRotor12D(x0, xg)
    
    # Create animation
    # print("Creating animation...")
    # anim = create_diffusion_animation(trajs_all, env, save_path=f'results/{date_str}_diffusion_animation.gif')
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajs[:, 0], trajs[:, 1], trajs[:, 2], '-b', label='Optimized via MBD')
    ax.scatter(data.x0[0], data.x0[1], data.x0[2], c='g', marker='o', label='Start Position')
    ax.scatter(data.goal[0], data.goal[1], data.goal[2], c='r', marker='*', label='Goal Position')
    ax.scatter(trajs[-1,0],trajs[-1,1],trajs[-1,2], c='y', marker='x', label='Final Position')
    env.plot_scenario(ax)
    # env.plot_dist_to_obs(ax,trajs)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig(f'results/{date_str}_mbd_quadrotor_randomNoise.png')
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(control_seq[:,0], label='tau_1')
    ax1.plot(control_seq[:,1], label='tau_2')
    ax1.plot(control_seq[:,2], label='tau_3')
    ax1.plot(control_seq[:,3], label='f_z (thrust)')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Denoised Control Input Sequence')
    ax1.legend()
    plt.savefig(f'results/{date_str}_mbd_control_seq.png')
    plt.show()
    
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    ax2.plot(trajs[:,0], label='x')
    ax2.plot(trajs[:,1], label='y')
    ax2.plot(trajs[:,2], label='z')
    ax2.plot(trajs[:,3], label='psi')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('State Trajectory via Roll-out')
    ax2.legend()
    plt.savefig(f'results/{date_str}_mbd_traj_components.png')
    plt.show()

    
    # plot drone's distance to each obstacle over the horizon
    fig = plt.figure()
    ax3 = fig.add_subplot(111)
    env.plot_dist_to_obs(trajs,ax3)
    plt.savefig(f'results/{date_str}_mbd_dist_to_obs.png')
    plt.show()


    