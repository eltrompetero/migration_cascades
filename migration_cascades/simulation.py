import numpy as np
from itertools import chain
from numba import njit


@njit
def flip(J, field, state, n1_res):
    """Flip function for spin equilibration based on energy considerations."""
    # Simple Ising-like flip probability based on field and coupling
    energy_diff = 2 * field * (2 * state - 1)  # energy difference for flip
    # Add coupling contribution (mean field approximation)
    coupling_contrib = 2 * J * (2 * state - 1) * (2 * n1_res - 1)
    total_energy_diff = energy_diff + coupling_contrib
    
    if total_energy_diff <= 0:
        return True
    else:
        return np.random.rand() < np.exp(-total_energy_diff)


@njit
def rand_choice(options, probabilities):
    """Random choice from options with given probabilities."""
    cumsum = np.cumsum(probabilities)
    rand_val = np.random.rand()
    for i in range(len(cumsum)):
        if rand_val <= cumsum[i]:
            return options[i]
    return options[-1]


def m_per_reservoir(S, R):
    """Calculate magnetization per reservoir."""
    magnetizations = np.zeros(R)
    for r in range(R):
        reservoir_spins = S[S[:, 2] == r, 1]
        if len(reservoir_spins) > 0:
            magnetizations[r] = np.mean(2 * reservoir_spins - 1)  # Convert 0,1 to -1,1
    return magnetizations


class CoupledReservoirs:
    """
    A class representing a system of coupled spin reservoirs with transfer dynamics.
    
    This implements a model where spins can equilibrate within reservoirs and transfer
    between reservoirs according to specified rates.
    """
    
    def __init__(self, n0=100, el=2.0, g=100.0, h_mu=-10, h_del=5, J=22., R=8, dt=1e-5,
                 rng=None):
        """
        Initialize the coupled reservoirs system.
        
        Parameters:
        -----------
        n0 : int
            Number of starting spins in each reservoir
        el : float
            Mean of transfer rate distribution
        g : float
            Equilibration rate
        h_mu : float
            Mean of field distribution
        h_del : float
            Half-width of field distribution
        J : float
            Coupling strength. This will be normalized by n0.
        R : int
            Number of reservoirs
        dt : float
            Time step
        rng : numpy.random.Generator, optional
            Random number generator for reproducibility
        """
        self.n0 = n0
        self.el = el
        self.g = g
        self.h_mu = h_mu
        self.h_del = h_del
        self.R = R
        self.dt = dt
        self.J = J / n0  # coupling strength
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Initialize state variables
        self.S = None
        self.L = None
        self.p_eq = None
        self.p_L = None
        self.n_res = None
        self.n1_res = None
        self.res_numbers = None
        
    def setup(self):
        """
        Set up the initial state of the system.
        """
        # Initialize transfer rate matrix
        self.L = np.zeros((self.R, self.R)) + self.el
        np.fill_diagonal(self.L, 0.)
        
        # Initialize spin array: col0: field, col1: state, col2: reservoir
        self.S = np.zeros((self.n0 * self.R, 3))
        self.S[:, 0] = self.rng.choice([self.h_mu - self.h_del, self.h_mu + self.h_del], 
                                 size=self.n0 * self.R)
        self.S[:, 1] = self.rng.choice([0, 1], size=self.n0 * self.R)
        self.S[:, 2] = list(chain.from_iterable([list(range(self.R)) for i in range(self.n0)]))
        
        # Set asymmetric starting conditions for reservoirs
        for r in range(self.R):
            self.S[:, 1][self.S[:, 2] == r] = self.rng.choice([0, 1])
        
        # Calculate transition probabilities given dt
        self.p_eq = self.dt * self.g
        self.p_L = self.dt * self.L
        self.p_L[np.diag_indices_from(self.L)] = 1 - self.p_L.sum(0)
        
        # Initialize reservoir counters
        self.n_res = np.zeros(self.R, dtype=np.int32) + self.n0
        self.n1_res = np.zeros(self.R, dtype=np.int32)
        for r in range(self.R):
            self.n1_res[r] += self.S[:, 1][self.S[:, 2] == r].sum()
        
        self.res_numbers = np.arange(self.R)
    
    def _loop_step(self, runtime):
        """
        Internal method for running the simulation loop.
        
        Parameters:
        -----------
        runtime : float
            Duration to run the simulation
            
        Returns:
        --------
        tuple
            Updated (S, n1_res, n_res) arrays
        """
        return self._numba_loop(runtime, self.S, self.n1_res, self.n_res, 
                                self.res_numbers, self.p_eq, self.p_L, self.J, 
                                self.n0, self.R, self.dt)
    
    @staticmethod
    @njit
    def _numba_loop(runtime, S, n1_res, n_res, res_numbers, p_eq, p_L, J, n0, R, dt):
        """
        Numba-compiled simulation loop for performance.
        """
        total_t = 0.
        while total_t < runtime:
            # One iteration
            rand_spin_ix = np.random.randint(n0 * R)
            origin_res = int(S[rand_spin_ix, 2])
            
            # Either equilibrate or transfer
            if np.random.rand() < p_eq:
                if flip(J, S[rand_spin_ix, 0], S[rand_spin_ix, 1], n1_res[origin_res]):
                    S[rand_spin_ix, 1] = 1 - S[rand_spin_ix, 1]
                    if S[rand_spin_ix, 1]:
                        n1_res[origin_res] += 1
                    else:
                        n1_res[origin_res] -= 1
            else:  # transfer
                dest_ix = rand_choice(res_numbers, p_L[:, origin_res])
                if dest_ix != origin_res:
                    if S[rand_spin_ix, 1]:
                        n1_res[origin_res] -= 1
                        n1_res[dest_ix] += 1
                    n_res[origin_res] -= 1
                    n_res[dest_ix] += 1
                    # Update spin's location
                    S[rand_spin_ix, 2] = dest_ix
            
            total_t += dt
        return S, n1_res, n_res
    
    def run(self, n_samples=200, sample_dt=5.0):
        """
        Run the simulation and measure magnetization.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to collect
        sample_dt : float
            Time interval between samples
            
        Returns:
        --------
        numpy.ndarray
            Array of magnetization values over time (shape: n_samples+1 x R)
        """
        if self.S is None:
            raise RuntimeError("System not initialized. Call setup() first.")
        
        m_t = np.zeros((n_samples + 1, self.R))
        m_t[0] = m_per_reservoir(self.S, self.R)
        
        for i in range(n_samples):
            self.S, self.n1_res, self.n_res = self._loop_step(sample_dt)
            m_t[i + 1] = m_per_reservoir(self.S, self.R)
        
        return m_t
    
    def get_state(self):
        """
        Get the current state of the system.
        
        Returns:
        --------
        dict
            Dictionary containing current system state
        """
        return {
            'spins': self.S.copy() if self.S is not None else None,
            'n_res': self.n_res.copy() if self.n_res is not None else None,
            'n1_res': self.n1_res.copy() if self.n1_res is not None else None,
            'magnetizations': m_per_reservoir(self.S, self.R) if self.S is not None else None
        }
    
    def reset(self):
        """Reset the system to initial conditions."""
        self.setup()
    
    def set_transfer_rates(self, L_matrix):
        """
        Set custom transfer rate matrix.
        
        Parameters:
        -----------
        L_matrix : numpy.ndarray
            R x R transfer rate matrix
        """
        if L_matrix.shape != (self.R, self.R):
            raise ValueError(f"Transfer matrix must be {self.R}x{self.R}")
        
        self.L = L_matrix.copy()
        np.fill_diagonal(self.L, 0.)
        self.p_L = self.dt * self.L
        self.p_L[np.diag_indices_from(self.L)] = 1 - self.p_L.sum(0)


# Example usage
if __name__ == "__main__":
    # Create and run simulation
    reservoirs = CoupledReservoirs(n0=100, R=8, rng=np.random.RandomState(42))
    reservoirs.setup()
    
    # Run simulation
    magnetizations = reservoirs.run(n_samples=200, sample_dt=5.0)
    
    print(f"Final magnetizations: {magnetizations[-1]}")
    print(f"Magnetization shape: {magnetizations.shape}")