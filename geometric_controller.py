import jax.numpy as jnp
import numpy as np
from drax.systems.quadrotor import QuadrotorOCP
import matplotlib.pyplot as plt

prob = QuadrotorOCP(horizon=50, x_init=jnp.zeros(12))


class DirtyDerivative:
    """
    Python equivalent of the MATLAB DirtyDerivative class.
    Provides a causal, band-limited differentiator with transfer function:
        P(s) = s / (tau + s)
    discretized via Tustin's method.
    """
    def __init__(self, order: int, tau: float, Ts: float):
        """
        order: how many initial samples to skip before differentiation
        tau:   time constant of the low-pass filter
        Ts:    sampling interval
        """
        self.order = order
        self.tau = tau
        self.Ts = Ts
        
        # Discretized filter coefficients
        self.a1 = (2 * tau - Ts) / (2 * tau + Ts)
        self.a2 = 2.0 / (2 * tau + Ts)
        
        # Internal state
        self.dot = None    # current derivative estimate
        self.x_prev = None # previous input
        self.it = 0       # iteration counter

    def calculate(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the filtered derivative of input x.
        x can be a scalar or array; returns an array of same shape.
        """
        x_arr = np.asarray(x)
        x_vec = x_arr.ravel()

        if self.it == 0:
            # initialize on first call
            self.dot = np.zeros_like(x_vec)
            self.x_prev = np.zeros_like(x_vec)

        if self.it > self.order:
            # apply differentiator
            self.dot = self.a1 * self.dot + self.a2 * (x_vec - self.x_prev)

        self.it += 1
        self.x_prev = x_vec

        return self.dot.reshape(x_arr.shape)



def closed_loop_geometric_control(x_traj, goal_pos_abs, trajectory_derivatives=None):
    """
    Geometric SE(3) controller with analytic desired derivatives and model-based accel.
    - Desired x,y,z and their time-derivatives come from `trajectory_derivatives`.
    - Actual acceleration uses vdot = g*e3 - (f/m) * (R @ e3) computed AFTER thrust f is known.
    - Uses tangent heading psi_d = atan2(vy, vx) for b1d, db1d/dt, d2b1d/dt2.
    Assumes the desired arrays are aligned with x_traj in time (same dt, same length).
    """
    import numpy as np
    import jax.numpy as jnp

    def vee_map(M):
        return jnp.array([M[2,1], M[0,2], M[1,0]])

    def rot_x(phi):
        c,s = jnp.cos(phi), jnp.sin(phi)
        return jnp.array([[1,0,0],[0,c,-s],[0,s,c]])

    def rot_y(theta):
        c,s = jnp.cos(theta), jnp.sin(theta)
        return jnp.array([[ c,0, s],[0,1,0],[-s,0,c]])

    def rot_z(psi):
        c,s = jnp.cos(psi), jnp.sin(psi)
        return jnp.array([[ c,-s,0],[ s, c,0],[0,0,1]])

    def hat(v):
        return jnp.array([[0.0, -v[2],  v[1]],
                          [v[2],  0.0, -v[0]],
                          [-v[1], v[0], 0.0]])

    use_analytical = trajectory_derivatives is not None

    # Plant / constants
    m, g = prob.m, prob.g
    Ix, Iy, Iz = prob.Ix, prob.Iy, prob.Iz
    J = jnp.array([[Ix, 0, 0],[0, Iy, 0],[0, 0, Iz]])
    dt = prob.dt
    e3 = jnp.array([0.0, 0.0, 1.0])

    # Gains
    kx_gain = 15*m
    kv_gain = 5*m
    kR_gain = 3
    kOmega_gain = 2

    # kx_gain = 15*m
    # kv_gain = 5*m
    # kR_gain = 5
    # kOmega_gain = 1

    
    # Buffers
    T = x_traj.shape[0]
    xs = np.array(x_traj)
    state_sim = jnp.zeros((T, 12))
    control_sim = jnp.zeros((T-1, 4))
    
    # Init state with first row of x_traj (angles, etc., as you did)
    state_sim = state_sim.at[0].set(x_traj[0])
    
    # f_list = []
    M_arrays = np.zeros((T-1,3))
    # Main loop
    for i in range(T-1):
        idx = i  # use current-time desired signals
    
        # Desired position and derivatives (analytic if provided, else zeros)
        xd = xs[idx, :3]
        if use_analytical:
            xd_1dot = jnp.array([trajectory_derivatives['x_dot'][idx],
                                 trajectory_derivatives['y_dot'][idx],
                                 trajectory_derivatives['z_dot'][idx]])
            xd_2dot = jnp.array([trajectory_derivatives['x_ddot'][idx],
                                 trajectory_derivatives['y_ddot'][idx],
                                 trajectory_derivatives['z_ddot'][idx]])
            xd_3dot = jnp.array([trajectory_derivatives['x_dddot'][idx],
                                 trajectory_derivatives['y_dddot'][idx],
                                 trajectory_derivatives['z_dddot'][idx]])
            xd_4dot = jnp.array([trajectory_derivatives['x_ddddot'][idx],
                                 trajectory_derivatives['y_ddddot'][idx],
                                 trajectory_derivatives['z_ddddot'][idx]])

            psi_des = jnp.array([trajectory_derivatives['psi'][idx]])
            psi_dot = jnp.array([trajectory_derivatives['psi_dot'][idx]])
            psi_ddot = jnp.array([trajectory_derivatives['psi_ddot'][idx]])
            tau_filter = 0.05
            Ts = prob.dt
            dv1dt = DirtyDerivative(1,tau_filter, Ts)
            dv2dt = DirtyDerivative(2,tau_filter*10, Ts)
        else:
            xd_1dot = jnp.zeros(3)
            xd_2dot = jnp.zeros(3)
            xd_3dot = jnp.zeros(3)
            xd_4dot = jnp.zeros(3)

        c, s = float(np.cos(psi_des)), float(np.sin(psi_des))
        b1d      = jnp.array([ c,  s, 0.0])
        b1d_1dot = jnp.array([-s,  c, 0.0]) * psi_dot
        b1d_2dot = jnp.array([-c, -s, 0.0]) * (psi_dot**2) + jnp.array([-s,  c, 0.0]) * psi_ddot

        # Current (actual) state
        x = state_sim[i, 0:3]
        v = state_sim[i, 6:9]
        psi, theta, phi = state_sim[i, 3], state_sim[i, 4], state_sim[i, 5]
        Omega = state_sim[i, 9:12]
        R = rot_z(psi) @ rot_y(theta) @ rot_x(phi)

        # Tracking errors that do NOT need actual accel/jerk
        ex = x - xd
        ev = v - xd_1dot

        # --- Compute translational control (A, f) FIRST ---
        A = -kx_gain*ex - kv_gain*ev + m*g*e3 + m*xd_2dot
        f = jnp.dot(A, R @ e3)
        f = jnp.clip(f, 0.0, 20.0)
        
        # --- Now model-based actual acceleration/jerk (no algebraic loop) ---
        v_1dot = -g*e3 + (float(f)/m) * (R @ e3)     # clean, low-lag
        # v_2dot = jnp.zeros(3)                       # start with zero (optional: light filter)
        v_2dot = dv2dt.calculate(v_1dot)
        # Higher-order error terms (optional but now causal)
        ea = v_1dot - xd_2dot
        ej = v_2dot - xd_3dot

        # Build desired attitude from A
        A_norm = jnp.linalg.norm(A)
        if float(A_norm) < 1e-6:
            b3c = e3
        else:
            b3c = A / A_norm

        C = jnp.cross(b3c, b1d)
        C_norm = jnp.linalg.norm(C)
        if float(C_norm) < 1e-6:
            # pick any vector not parallel to b3c
            temp = jnp.array([0.0, 1.0, 0.0])
            C = jnp.cross(b3c, temp)
            C_norm = jnp.linalg.norm(C)
        b2c = C / C_norm
        b1c = jnp.cross(b2c, b3c)

        Rc = jnp.column_stack([b1c, b2c, b3c])

        # Time derivatives of body axes
        # Ȧ uses the causal ea just computed
        A_1dot = -kx_gain*ev - kv_gain*ea + m*xd_3dot

        # d/dt(b3c) from normalized A
        # b3c = A/||A||
        # b3c_dot = Ȧ/||A|| - (A^T Ȧ)/||A||^3 * A
        A_dot_proj = jnp.dot(A, A_1dot)
        b3c_1dot = A_1dot / A_norm - (A_dot_proj / (A_norm**3)) * A

        C_1dot = jnp.cross(b3c_1dot, b1d) + jnp.cross(b3c, b1d_1dot)
        C_dot_proj = jnp.dot(C, C_1dot)
        b2c_1dot = C_1dot / C_norm - (C_dot_proj / (C_norm**3)) * C
        b1c_1dot = jnp.cross(b2c_1dot, b3c) + jnp.cross(b2c, b3c_1dot)

        # Second derivatives
        # Ä uses ej (causal)
        A_2dot = -kx_gain*ea - kv_gain*ej + m*xd_4dot

        term1 = A_2dot / A_norm
        term2 = -(2.0 / (A_norm**3)) * A_dot_proj * A_1dot
        term3 = -((jnp.linalg.norm(A_1dot)**2 + jnp.dot(A, A_2dot)) / (A_norm**3)) * A
        term4 = (3.0 / (A_norm**5)) * (A_dot_proj**2) * A
        b3c_2dot = term1 + term2 + term3 + term4

        C_2dot = jnp.cross(b3c_2dot, b1d) + jnp.cross(b3c, b1d_2dot) + 2*jnp.cross(b3c_1dot, b1d_1dot)

        term5 = C_2dot / C_norm
        term6 = -(2.0 / (C_norm**3)) * jnp.dot(C, C_1dot) * C_1dot
        term7 = -((jnp.linalg.norm(C_1dot)**2 + jnp.dot(C, C_2dot)) / (C_norm**3)) * C
        term8 = (3.0 / (C_norm**5)) * (jnp.dot(C, C_1dot)**2) * C
        b2c_2dot = term5 + term6 + term7 + term8

        b1c_2dot = jnp.cross(b2c_2dot, b3c) + jnp.cross(b2c, b3c_2dot) + 2*jnp.cross(b2c_1dot, b3c_1dot)

        # Desired angular velocity/acceleration
        Rc_1dot = jnp.column_stack([b1c_1dot, b2c_1dot, b3c_1dot])
        Rc_2dot = jnp.column_stack([b1c_2dot, b2c_2dot, b3c_2dot])

        Omegac = vee_map(Rc.T @ Rc_1dot)
        Omegac_1dot = vee_map(Rc.T @ Rc_2dot - hat(Omegac) @ hat(Omegac))

        # Attitude errors
        eR = vee_map(0.5*(Rc.T @ R - R.T @ Rc))
        eOmega = Omega - R.T @ Rc @ Omegac

        # Moments
        M = -kR_gain*eR - kOmega_gain*eOmega + jnp.cross(Omega, J @ Omega)  - J @ (hat(Omega) @ (R.T @ Rc @ Omegac) - R.T @ Rc @ Omegac_1dot)
        M = M * 0.1  
        M = jnp.clip(M, -5, 5)
        M_arrays[i,:] = M

        # Control vector [Mx, My, Mz, f]
        u_current = jnp.concatenate([M, jnp.array([f])])
        control_sim = control_sim.at[i, :].set(u_current)

        # Step the plant
        x_next = prob.dynamics(state_sim[i], control_sim[i])

        # Safety clamp
        if float(x_next[2]) < 0.05:
            print(f"WARNING: ground collision at step {i}, correcting height")

        state_sim = state_sim.at[i+1, :].set(x_next)
    
    # Metrics
    pos_sq_err = jnp.sum((state_sim[:, 0:3] - xs[:, 0:3])**2, axis=1)
    mse = jnp.mean(pos_sq_err)
    print(f"Mean squared position tracking error: {mse:.4f} m²")

    final_pos_err = jnp.linalg.norm(state_sim[-1, 0:3] - goal_pos_abs)
    print(f"Final closed loop position error : {final_pos_err:.4f} m")

    #optinal:
    # plt.plot(M_arrays[:,0], label='tau_x')
    # plt.plot(M_arrays[:,1], label='tau_y')
    # plt.plot(M_arrays[:,2], label='tau_z')
    # plt.legend()
    # plt.show()
    

    return final_pos_err, mse, state_sim




# def closed_loop_geometric_control(x_traj, goal_pos_abs, trajectory_derivatives=None):
#     """
#     Closed-loop geometric controller for quadrotor trajectory tracking.
    
#     Args:
#         x_traj: Trajectory to track
#         goal_pos_abs: Absolute goal position
#         trajectory_derivatives: Dictionary containing analytical derivatives of the trajectory
#                                If None, will use numerical derivatives
#     """
#     def vee_map(M):
#             # M is 3×3 skew; return the vector in R^3
#             return jnp.array([M[2,1], M[0,2], M[1,0]])
    
#     def rot_x(phi):
#         c,s = jnp.cos(phi), jnp.sin(phi)
#         return jnp.array([[1,0,0],
#                         [0, c,-s],
#                         [0, s, c]])

#     def rot_y(theta):
#         c,s = jnp.cos(theta), jnp.sin(theta)
#         return jnp.array([[ c,0, s],
#                         [ 0,1, 0],
#                         [-s,0, c]])

#     def rot_z(psi):
#         c,s = jnp.cos(psi), jnp.sin(psi)
#         return jnp.array([[ c,-s,0],
#                         [ s, c,0],
#                         [ 0, 0,1]])

#     def hat(v):
#         """
#         Convert a 3-vector v = [v1, v2, v3] into the 3×3 skew-symmetric matrix:
#             [  0   -v3   v2 ]
#             [  v3    0  -v1 ]
#             [ -v2   v1    0 ]
#         """
#         return jnp.array([
#             [    0. , -v[2],  v[1]],
#             [ v[2],     0. , -v[0]],
#             [-v[1],  v[0],     0. ]
#         ])
        
#     # Check if analytical derivatives are provided
#     use_analytical_derivatives = trajectory_derivatives is not None
#     tau_filter = 0.05
#     Ts = prob.dt

#     # If using numerical derivatives, set up filters
#     if not use_analytical_derivatives:
#         # --- Instantiate filters for derivatives ---

#         # Create filters for position, velocity, and orientation derivatives
#         dx1dt = DirtyDerivative(1,tau_filter, Ts)
#         dx2dt = DirtyDerivative(2,tau_filter*10, Ts)
#         dx3dt = DirtyDerivative(3,tau_filter*10, Ts)
#         dx4dt = DirtyDerivative(4,tau_filter*10, Ts)
#         db1dt = DirtyDerivative(1,tau_filter, Ts)
#         db2dt = DirtyDerivative(2,tau_filter*10, Ts)


#     dv1dt = DirtyDerivative(1,tau_filter, Ts)
#     dv2dt = DirtyDerivative(2,tau_filter*10, Ts)

#     # Simulate the closed-loop system
#     state_sim = jnp.zeros((x_traj.shape[0], 12))
#     control_sim = jnp.zeros((x_traj.shape[0]-1, 4))

#     # Initialize with the first state
#     m = prob.m
#     g = prob.g

#     x_init = x_traj[0,:] 
#     # Set initial state
#     state_sim = state_sim.at[0].set(x_init)
#     #nlp_state: (12, 50)
#     #xs : (50, 12)

#     T = x_traj.shape[0]
#     dt = prob.dt

#     # Get inertia matrix
#     Ix, Iy, Iz = prob.Ix, prob.Iy, prob.Iz  # moments of inertia
#     J = jnp.array([[Ix, 0, 0],
#                     [0, Iy, 0],
#                     [0, 0, Iz]])  # inertia matrix

#     # Control gains
#     # gains "optimized" for our method
#     kx_gain = 26*m   # Position gain
#     kv_gain = 10*m    # Velocity gain  
#     kR_gain = 10
#     kOmega_gain = 2


#     # # gains taken from the geometric control paper:
#     # kx_gain = 16*m 
#     # kv_gain = 5.6*m 
#     # kR_gain = 8.81
#     # kOmega_gain = 2.54

#     # Inertial frame z-axis
#     e3 = jnp.array([0, 0, 1])
    

#     xs = np.array(x_traj) 
#     # print(f'xs has shape {xs.shape}')
#     for i in range(T-1):
#         xd = xs[i+1,:3] #desired trajectory generated by diffusion solver after post-processing
#         # Desired body-1 axis (corresponds to desired yaw)
#         # psi_des = 0
#         psi_des = jnp.arctan2(xd[1], xd[0])
#         b1d = jnp.array([jnp.cos(psi_des), jnp.sin(psi_des), 0.0])
        
#         # Extract current state
#         x = state_sim[i, 0:3]  # position [x, y, z]
#         v = state_sim[i, 6:9]  # velocity [vx, vy, vz]
#         psi, theta, phi = state_sim[i, 3], state_sim[i, 4], state_sim[i, 5]  # yaw, pitch, roll
#         Omega = state_sim[i, 9:12]  # angular velocity [p, q, r]
        
#         # Construct current rotation matrix R from Euler angles (ZYX convention)
#         R = rot_z(psi) @ rot_y(theta) @ rot_x(phi)
        
#         # Get derivatives of desired position - either analytical or numerical
#         if use_analytical_derivatives:
#             # Use analytical derivatives from the spline
#             idx = i + 1  # Current index in the trajectory
            
#             # Position derivatives
#             xd_1dot = np.array([trajectory_derivatives['x_dot'][idx], 
#                                trajectory_derivatives['y_dot'][idx], 
#                                trajectory_derivatives['z_dot'][idx]])  # Velocity
            
#             xd_2dot = np.array([trajectory_derivatives['x_ddot'][idx], 
#                                trajectory_derivatives['y_ddot'][idx], 
#                                trajectory_derivatives['z_ddot'][idx]])  # Acceleration
            
#             xd_3dot = np.array([trajectory_derivatives['x_dddot'][idx], 
#                                trajectory_derivatives['y_dddot'][idx], 
#                                trajectory_derivatives['z_dddot'][idx]])  # Jerk
            
#             xd_4dot = np.array([trajectory_derivatives['x_ddddot'][idx], 
#                                trajectory_derivatives['y_ddddot'][idx], 
#                                trajectory_derivatives['z_ddddot'][idx]])  # Snap
            
#             # For body-1 axis derivatives, we still need to compute them
#             # Since they depend on the yaw angle which is computed from position
#             eps = 1e-6
#             vx, vy = float(xd_1dot[0]), float(xd_1dot[1])
#             ax, ay = float(xd_2dot[0]), float(xd_2dot[1])
#             psi_des = np.arctan2(vy, vx)
#             den_v = vx*vx + vy*vy + eps
#             psi_dot = (vx*ay - vy*ax) / den_v
#             jx, jy = float(xd_3dot[0]), float(xd_3dot[1])
#             N = vx*ay - vy*ax
#             dN = vx*jy - vy*jx
#             D = den_v
#             dD = 2.0*(vx*ax + vy*ay)
#             psi_ddot = (dN*D - N*dD) / (D*D)

#             c, s = float(np.cos(psi_des)), float(np.sin(psi_des))
#             b1d      = np.array([ c,  s, 0.0])
#             b1d_1dot = np.array([-s,  c, 0.0]) * psi_dot
#             b1d_2dot = np.array([-c, -s, 0.0]) * (psi_dot**2) + np.array([-s,  c, 0.0]) * psi_ddot
                
#             # For current velocity derivatives, we can use the analytical ones for the desired trajectory
#             # but we still need numerical derivatives for the actual state feedback
#             v_1dot = dv1dt.calculate(v)  # Acceleration
#             v_2dot = dv2dt.calculate(v_1dot)  # Jerk
#         else:
#             # Use numerical derivatives with filters
#             xd_1dot = dx1dt.calculate(xd)  # Velocity
#             xd_2dot = dx2dt.calculate(xd_1dot)  # Acceleration
#             xd_3dot = dx3dt.calculate(xd_2dot)  # Jerk
#             xd_4dot = dx4dt.calculate(xd_3dot)  # Snap
            
#             # Numerical derivatives of desired body-1 axis (the yaw vector)
#             b1d_1dot = db1dt.calculate(b1d)
#             b1d_2dot = db2dt.calculate(b1d_1dot)
            
#             # Numerical derivatives of current velocity
#             v_1dot = dv1dt.calculate(v)  # Acceleration
#             v_2dot = dv2dt.calculate(v_1dot)  # Jerk
        
#         # Calculate errors
#         ex = x - xd
#         ev = v - xd_1dot
#         ea = v_1dot - xd_2dot
#         ej = v_2dot - xd_3dot
        
#         # Ideal thrust vector (world frame) to stabilize translational motion
#         # Note: Adjusting for z-axis pointing up instead of down
#         A = -kx_gain*ex - kv_gain*ev + m*g*e3 + m*xd_2dot
        
#         # Calculate thrust (dot product with body z-axis)
#         # Note: In your implementation, thrust is positive upward
#         f = jnp.dot(A, R @ e3)
        
#         # Normalized feedback function
#         A_norm = jnp.linalg.norm(A)
#         if A_norm < 1e-6:
#             b3c = e3  # Default to upward if A is near zero
#         else:
#             b3c = A / A_norm
        
#         # Construct b1c (first desired body axis)
#         C = jnp.cross(b3c, b1d)
#         C_norm = jnp.linalg.norm(C)
        
        
#         if C_norm < 1e-6:
#             # Handle singularity case
#             temp = jnp.array([0.0, 1.0, 0.0])
#             C = jnp.cross(b3c, temp)
#             C_norm = jnp.linalg.norm(C)
        
#         b2c = C / C_norm
#         b1c = jnp.cross(b2c, b3c)
        
#         # Collect desired rotation matrix relative to world frame
#         Rc = jnp.column_stack([b1c, b2c, b3c])
        
#         # Time derivatives of body axes
#         A_1dot = -kx_gain*ev - kv_gain*ea + m*xd_3dot
#         b3c_1dot = A_1dot/A_norm - (jnp.dot(A, A_1dot)/A_norm**3)*A
        
#         C_1dot = jnp.cross(b3c_1dot, b1d) + jnp.cross(b3c, b1d_1dot)
#         b2c_1dot = C_1dot/C_norm - (jnp.dot(C, C_1dot)/C_norm**3)*C
#         b1c_1dot = jnp.cross(b2c_1dot, b3c) + jnp.cross(b2c, b3c_1dot)
        
#         # Second time derivatives of body axes
#         A_2dot = -kx_gain*ea - kv_gain*ej + m*xd_4dot
#         # A_2dot = -kx_gain*ea - kv_gain*ej
        
#         term1 = A_2dot/A_norm
#         term2 = -(2.0/A_norm**3)*jnp.dot(A, A_1dot)*A_1dot
#         term3 = -((jnp.linalg.norm(A_1dot)**2 + jnp.dot(A, A_2dot))/A_norm**3)*A
#         term4 = (3.0/A_norm**5)*(jnp.dot(A, A_1dot)**2)*A
#         b3c_2dot = term1 + term2 + term3 + term4
        
#         C_2dot = jnp.cross(b3c_2dot, b1d) + jnp.cross(b3c, b1d_2dot) + 2*jnp.cross(b3c_1dot, b1d_1dot)
        
#         term5 = C_2dot/C_norm
#         term6 = -(2.0/C_norm**3)*jnp.dot(C, C_1dot)*C_1dot
#         term7 = -((jnp.linalg.norm(C_1dot)**2 + jnp.dot(C, C_2dot))/C_norm**3)*C
#         term8 = (3.0/C_norm**5)*(jnp.dot(C, C_1dot)**2)*C
#         b2c_2dot = term5 + term6 + term7 + term8
        
#         b1c_2dot = jnp.cross(b2c_2dot, b3c) + jnp.cross(b2c, b3c_2dot) + 2*jnp.cross(b2c_1dot, b3c_1dot)
        
#         # Extract calculated angular velocities and their time-derivatives
#         Rc_1dot = jnp.column_stack([b1c_1dot, b2c_1dot, b3c_1dot])
#         Rc_2dot = jnp.column_stack([b1c_2dot, b2c_2dot, b3c_2dot])
        
#         Omegac = vee_map(Rc.T @ Rc_1dot)
#         # print(f'desird angular velocity is {Omegac}')
#         Omegac_1dot = vee_map(Rc.T @ Rc_2dot - hat(Omegac) @ hat(Omegac))
#         # print(f'desired angular acceleration is {Omegac_1dot}')
        
#         # Calculate attitude error
#         eR = vee_map(0.5*(Rc.T @ R - R.T @ Rc))
#         eOmega = Omega - R.T @ Rc @ Omegac
        
#         # Calculate control moments
#         M = -kR_gain*eR - kOmega_gain*eOmega + jnp.cross(Omega, J @ Omega) - J @ (hat(Omega) @ R.T @ Rc @ Omegac - R.T @ Rc @ Omegac_1dot)
#         M = M * 0.1

#         # Combine moments and thrust
#         u_current = jnp.concatenate([M, jnp.array([f])])
        
#         control_sim = control_sim.at[i,:].set(u_current)
        
#         # Simulate one step forward using the quadrotor dynamics
#         x_next = prob.dynamics(state_sim[i], control_sim[i])
        
#         # Safety check - prevent negative height values
#         if x_next[2] < 0.05:  # If height is too low
#             print(f"WARNING: ground collision at step {i}, correcting height")

#         # Update state for next iteration
#         state_sim = state_sim.at[i+1,:].set(x_next)

#     # Calculate tracking error
#     # Calculate squared differences between actual and desired positions
#     position_squared_error = jnp.sum((state_sim[:, 0:3] - xs[:,0:3])**2, axis=1)
#     # Calculate mean squared error
#     mse = jnp.mean(position_squared_error)
    
#     print(f"Mean squared position tracking error: {mse:.4f} m²")

#     final_pos_err = jnp.linalg.norm(state_sim[-1, 0:3] - goal_pos_abs)
#     # Calculate final position error for compatibility with existing code
#     print(f'Final closed loop position error : {final_pos_err:.4f} m')
#     return final_pos_err, mse, state_sim