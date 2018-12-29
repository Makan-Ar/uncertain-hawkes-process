import numpy as np
import matplotlib.pyplot as plt
from tick.hawkes import SimuHawkesExpKernels, SimuPoissonProcess
from tick.plot import plot_point_process


# A uni-dimension Hawkes Process model, combined with a Poisson process as the uncertain events
class HawkesUncertainModel:
    n_nodes = 1 # fix Hawkes process dimension to 1
    run_time = None
    delta = None

    hawkes = None
    hawkes_exp = None
    poisson = None
    poisson_exp = None

    def __init__(self, h_lambda, h_alpha, h_beta, h_exp_beta, p_lambda, p_exp_beta, run_time=100, delta=0.4, seed=None):
        self.run_time = run_time
        self.delta = delta

        self.h_lambda = h_lambda
        self.h_alpha = h_alpha
        self.h_beta = h_beta
        self.p_lambda = p_lambda
        self.h_exp_beta = h_exp_beta
        self.p_exp_beta = p_exp_beta

        # Simulate Hawkes Process
        adjacency = self.h_alpha * np.ones((self.n_nodes, self.n_nodes))  # alpha (intensities)
        decays = self.h_alpha * np.ones((self.n_nodes, self.n_nodes))  # beta
        baseline = self.h_lambda * np.ones(self.n_nodes)  # Baseline intensities of Hawkes processes
        self.hawkes = SimuHawkesExpKernels(adjacency=adjacency, decays=decays, baseline=baseline, verbose=False,
                                           seed=seed)
        self.hawkes.end_time = self.run_time
        self.hawkes.track_intensity(self.delta)
        self.hawkes.simulate()

        # Simulate Poisson Process
        self.poisson = SimuPoissonProcess(self.p_lambda, end_time=run_time, verbose=False)
        self.poisson.simulate()

        # Simulate two exponential distribution as side information
        self.hawkes_exp = np.random.exponential(self.h_exp_beta, self.hawkes.n_total_jumps)
        self.poisson_exp = np.random.exponential(self.p_exp_beta, self.poisson.n_total_jumps)

    def plot_hawkes(self, n_points=50000, t_min=1, max_jumps=200):
        plot_point_process(self.hawkes, n_points=n_points, t_min=t_min, max_jumps=max_jumps)
        plt.show()

    def plot_poisson(self):
        plot_point_process(self.poisson)

    def plot_hawkes_uncertain(self):
        plt.scatter(self.hawkes.timestamps, self.hawkes_exp, c='blue',
                    label='Hawkes(alpha:{}, beta:{}, lambda:{}) w/ exp({})'.format(self.h_alpha, self.h_beta,
                                                                                   self.h_lambda, self.h_exp_beta))
        plt.scatter(self.poisson.timestamps, self.poisson_exp, c='red',
                    label="Poisson(lambda: {}) w/ exp({})".format(self.p_lambda, self.p_exp_beta))
        # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1, mode="expand", borderaxespad=0.)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("y = exp(mu)")
        plt.show()
