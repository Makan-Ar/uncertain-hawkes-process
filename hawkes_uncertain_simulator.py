import numpy as np
import matplotlib.pyplot as plt
from tick.hawkes import SimuHawkesExpKernels, SimuPoissonProcess
from tick.plot import plot_point_process


# A uni-dimension Hawkes Process model, combined with a Poisson process as the uncertain events
class HawkesUncertainModel:
    n_nodes = 1  # fix Hawkes process dimension to 1
    run_time = None
    delta = None

    hawkes = None
    hawkes_exp = None
    poisson = None
    poisson_exp = None

    mixed_expo = None
    mixed_timestamps = None
    mixed_labels = None
    noise_percentage = None

    def __init__(self, h_lambda, h_alpha, h_beta, h_exp_beta,
                 p_lambda, p_exp_beta,
                 run_time=100, delta=0.4, noise_percentage_ub=0.5, seed=None):
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

        sim_cnt = 0
        max_num_sim = 500
        while sim_cnt < max_num_sim:
            self.hawkes = SimuHawkesExpKernels(adjacency=adjacency, decays=decays, baseline=baseline, verbose=False,
                                               seed=seed)
            self.hawkes.end_time = self.run_time
            self.hawkes.track_intensity(self.delta)
            self.hawkes.simulate()

            # Simulate Poisson Process
            self.poisson = SimuPoissonProcess(self.p_lambda, end_time=run_time, verbose=False, seed=seed)
            self.poisson.simulate()

            # Accept a sim if neither of the sims are empty, and
            # no two events happen at the same time, and
            # noise_percentage_ub is not violated
            if (len(self.hawkes.timestamps[0]) > 0 and len(self.poisson.timestamps[0]) > 0 and
                len(np.intersect1d(self.hawkes.timestamps[0], self.poisson.timestamps[0], assume_unique=True)) == 0 and
                self.poisson.n_total_jumps / (self.poisson.n_total_jumps +
                                              self.hawkes.n_total_jumps) <= noise_percentage_ub):
                break
            elif seed is not None:
                exit('A valid simulation is not possible with the provided seed value.')

            sim_cnt += 1
        else:
            exit(f"After {max_num_sim} simulations, a valid simulation was not obtained. Try changing either poisson's "
                 f"rate or noise_percentage_ub.")

        self.mixed_timestamps = np.concatenate((self.hawkes.timestamps[0], self.poisson.timestamps[0]))
        sort_ind = np.argsort(self.mixed_timestamps)
        self.mixed_timestamps = self.mixed_timestamps[sort_ind]

        # Simulate two exponential distribution as side information
        self.hawkes_exp = np.random.exponential(1. / self.h_exp_beta, self.hawkes.n_total_jumps)
        self.poisson_exp = np.random.exponential(1. / self.p_exp_beta, self.poisson.n_total_jumps)

        # mixed exponential
        self.mixed_expo = np.concatenate((self.hawkes_exp, self.poisson_exp), axis=0)
        self.mixed_expo = self.mixed_expo[sort_ind]

        # mixed labels. Poisson = 1, Hawkes = 0
        self.mixed_labels = np.concatenate((np.zeros(len(self.hawkes_exp), dtype=int),
                                            np.ones(len(self.poisson_exp), dtype=int)), axis=0)
        self.mixed_labels = self.mixed_labels[sort_ind]

        self.noise_percentage = np.sum(self.mixed_labels) / len(self.mixed_labels)

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
