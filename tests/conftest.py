import pytest
import numpy as np


def generate_random_walk(n_steps, dim=3, diffusion_coeff=1.0, dt=1.0):
    """Generate a random walk trajectory for testing.
    
    Returns array of shape (n_steps+1, dim) with initial position at origin.
    """
    step_std = np.sqrt(2 * diffusion_coeff * dt)
    steps = np.random.normal(0, step_std, size=(n_steps, dim))
    trajectory = np.cumsum(steps, axis=0)
    return np.vstack([np.zeros((1, dim)), trajectory])


@pytest.fixture
def random_walk_file(tmp_path):
    """Create a temporary 3D random walk trajectory file (5000 steps)."""
    traj = generate_random_walk(n_steps=5000, dim=3, diffusion_coeff=0.1, dt=1.0)
    file_path = tmp_path / "test_traj.dat"
    np.savetxt(file_path, traj)
    return str(file_path)
