# PAD-TRO: Projection-Augmented Diffusion for Direct Trajectory Optimization 
[<img src="https://img.shields.io/badge/Backend-Jax-red.svg"/>](https://github.com/google/jax)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Install dependencies
To install dependencies: 
`pip install -e . `

## Open-loop trajectory generation
To generate trajectories using our algorithm, run `python3 state_diffusion_chronological_projection`

## (Optional) closed loop control
You may also use the provided `geometric_coontroller.py` to run closed-loop simulations for trajectory tracking. The controller is implemented based on this [paper](https://ieeexplore.ieee.org/document/5717652)

## Acknowledgement
Some parts of the code are adapted from [Model-Based Diffusion for Trajectory Optimization](https://github.com/LeCAR-Lab/model-based-diffusion/tree/main)


