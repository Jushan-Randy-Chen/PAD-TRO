# PAD-TRO: Projection-Augmented Diffusion for Direct Trajectory Optimization 
[<img src="https://img.shields.io/badge/Backend-Jax-red.svg"/>](https://github.com/google/jax)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This paper has been accepted for publication at the [2026 American Control Conference](https://acc2026.a2c2.org/)!
View the full paper here: [[Arxiv]](https://arxiv.org/abs/2510.04436)

Recent diffusion-based trajectory optimization frameworks rely on a single-shooting style approach where the denoised control sequence is applied to forward propagate the dynamical system, which cannot explicitly enforce constraints on the states and frequently leads to sub-optimal solutions. In this work, we propose a novel direct trajectory optimization approach via model-based diffusion, which directly generates a sequence of states. To ensure dynamic feasibility, we propose a gradient-free projection mechanism that is incorporated into the reverse diffusion process. Our results show that, compared to a recent state-of-the-art baseline, our approach leads to zero dynamic feasibility error and approximately 4x higher success rate in a quadrotor waypoint navigation scenario involving dense static obstacles.

## Install dependencies
To install dependencies: 
`pip install -e . `
### Warning: you need a JAX compatible GPU!!

## Open-loop trajectory generation
To generate trajectories using our algorithm, run `python3 state_diffusion_chronological_projection`

## (Optional) closed loop control
You may also use the provided `geometric_coontroller.py` to run closed-loop simulations for trajectory tracking. The controller is implemented based on this famous [paper](https://ieeexplore.ieee.org/document/5717652)

## Acknowledgement
Some parts of the code are adapted from [Model-Based Diffusion for Trajectory Optimization](https://github.com/LeCAR-Lab/model-based-diffusion/tree/main)


