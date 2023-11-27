import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

def compute_accelerations(positions, masses, G, epsilon):
    N = len(masses) 
    if len(positions) != N:
        raise ValueError("Length of positions must match the number of masses")
    accelerations = np.zeros((N, 3), dtype=np.float64)  # Initialise accelerations array
    
    # the pairwise acceleration due to gravity
    for i in range(N):
        for j in range(i+1, N):
            # Vector from body i to body j
            r_ij = positions[j] - positions[i]
            # Distance norm (magnitude of r_ij) with a small softening factor to prevent singularity
            r_ij_norm = np.linalg.norm(r_ij) + epsilon 
            # Gravitational force factor
            grav_force = G * masses[i] * masses[j] / r_ij_norm**3

            # Update acceleration of body i due to body j
            accelerations[i] += grav_force * r_ij / masses[i]
            # Update acceleration of body j due to body i
            accelerations[j] -= grav_force * r_ij / masses[j]

    return accelerations

def velocity_verlet(positions, velocities, masses, h, G, epsilon):
    N = len(masses) 
    if len(positions) != N:
        raise ValueError("Length of positions must match the number of masses")

    accelerations = compute_accelerations(positions, masses, G, epsilon)
    velocities_half = velocities + 0.5 * h * accelerations 
    new_positions  = positions + h * velocities_half
    new_accelerations = compute_accelerations(new_positions, masses, G, epsilon)
    new_velocities = velocities_half + 0.5 * h * new_accelerations

    return new_positions, new_velocities

def calculate_KE(velocities, masses):
    """
    Calculate the total kinetic energy of the system.
    KE = 0.5 * m * v^2 for each body
    """
    N = len(masses) 
    kinetic_energy = 0.0
    for i in range(N):
        v_i = velocities[i] 
        m_i = masses[i]   
        kinetic_energy += 0.5 * m_i * np.dot(v_i, v_i)
    return kinetic_energy

def calculate_PE(positions, masses, G, epsilon):
    """
    Calculate the total potential energy of the system.
    PE = -G * m1 * m2 / r for each unique pair of bodies
    """
    N = len(masses) 
    potential_energy = 0.0
    for i in range(N):
        for j in range(i+1, N): #calculate the interaction of body i with body j that comes after it in the list
            r_ij = np.linalg.norm(positions[i] - positions[j]) + epsilon
            potential_energy -= G * masses[i] * masses[j] / r_ij 

    return potential_energy

def compute_angular_momentum(positions, velocities, masses):
    """
    Calculate the total angular momentum of the system.
    L = r x p for each body, where p = m * v, and r x p is the cross product.
    """
    N = len(masses) 
    angular_momentum = np.zeros(3)
    for i in range(N):
        r_i = positions[i]
        v_i = velocities[i]
        p_i = masses[i] * v_i  # Linear momentum of body i
        angular_momentum += np.cross(r_i, p_i)
    
    modulus = np.linalg.norm(angular_momentum)
    return angular_momentum, modulus    

def run_simulation(positions, velocities, masses, h, num_steps, G, epsilon):
    """
    Run the n-body simulation.

    Parameters:
    - positions: Initial positions of the bodies.
    - velocities: Initial velocities of the bodies.
    - masses: Masses of the bodies.
    - h: Time step size.
    - num_steps: Number of steps in the simulation.

    Returns:
    - trajectories: Positions of the bodies at each time step.
    - energy_updated: Total energy of the system at each time step.
    - AM_mag_updated: Magnitude of angular momentum at each time step.
    """

    num_bodies = len(masses)
    trajectories = np.zeros((num_bodies, num_steps, 2))
    energy_updated = np.zeros(num_steps)
    AM_mag_updated = np.zeros(num_steps)

    # Simulation loop
    for step in range(num_steps):
        # Update positions and velocities
        positions, velocities = velocity_verlet(positions, velocities, masses, h, G, epsilon)
        
        # Calculate center of mass
        total_mass = np.sum(masses)
        COM = np.sum(positions.T * masses, axis=1) / total_mass
        # Subtract CoM from positions to correct for drift
        corrected_positions = positions - COM.reshape(-1, 3)
        for i in range(num_bodies):
            trajectories[i, step, :] = corrected_positions[i, :2]

        # Calculate and store energy and angular momentum
        KE = calculate_KE(velocities, masses)
        PE = calculate_PE(positions, masses, G, epsilon)
        total_E = KE + PE
        angular_momentum_vector, angular_momentum_modulus = compute_angular_momentum(positions, velocities, masses)
        energy_updated[step] = total_E
        AM_mag_updated[step] = angular_momentum_modulus

    return trajectories[:, :, :2], energy_updated, AM_mag_updated

def plot_simulation(trajectories, energy_updated, AM_mag_updated, names, num_steps, h):
    num_bodies = trajectories.shape[0]
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), dpi=300)

    # if body names are provided, otherwise use default names
    if not names:
        names = [f'Body {i+1}' for i in range(num_bodies)]

    # Trajectories plot
    for i in range(num_bodies):
        axs[0].plot(trajectories[i, :, 0], trajectories[i, :, 1], label = names[i])
    axs[0].scatter(0, 0, color='red', s=50, label='Center of Mass')
    axs[0].set_title('Orbit Trajectories')
    axs[0].set_xlabel('x position (m)')
    axs[0].set_ylabel('y position (m)')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].axis('equal')

    # Total energy plot
    initial_energy = energy_updated[0]
    relative_energy_change = ((energy_updated - initial_energy) / initial_energy) * 100
    time_array = np.arange(num_steps) * h
    axs[1].plot(time_array, relative_energy_change)
    axs[1].set_title('Relative Percentage Change in Energy Over Time')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Relative Change (%)')

    # Angular momentum plot
    axs[2].plot(time_array, AM_mag_updated)
    axs[2].set_title('Angular Momentum Over Time')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Angular Momentum Magnitude')

    plt.tight_layout()
    plt.show()