import numpy as np
import matplotlib.pyplot as plt
from tick.hawkes import SimuPoissonProcess
from tick.plot import plot_point_process


class UncertainPoissonModel:
    run_time = None
    delta = None

    poisson_1 = None
    poisson_exp_1 = None

    poisson_2 = None
    poisson_exp_2 = None

    def __init__(self, lambda_1, exp_beta_1, lambda_2, exp_beta_2, run_time=100, delta=0.4, seed_1=None, seed_2=None):
        self.run_time = run_time
        self.delta = delta

        self.lambda_1 = lambda_1
        self.exp_beta_1 = exp_beta_1
        self.lambda_2 = lambda_2
        self.exp_beta_2 = exp_beta_2

        # Simulate the first Poisson Process
        self.poisson_1 = SimuPoissonProcess(self.lambda_1, end_time=run_time, verbose=False, seed=seed_1)
        self.poisson_1.simulate()

        # Simulate the second Poisson Process
        self.poisson_2 = SimuPoissonProcess(self.lambda_2, end_time=run_time, verbose=False, seed=seed_2)
        self.poisson_2.simulate()

        # Simulate two exponential distribution as side information
        self.poisson_exp_1 = np.random.exponential(self.exp_beta_1, self.poisson_1.n_total_jumps)
        self.poisson_exp_2 = np.random.exponential(self.exp_beta_2, self.poisson_2.n_total_jumps)

    def plot_poisson_1(self):
        plot_point_process(self.poisson_1)

    def plot_poisson_2(self):
        plot_point_process(self.poisson_2)

    def plot_poisson_uncertain(self):
        plt.scatter(self.poisson_1.timestamps, self.poisson_exp_1, c='red',
                    label="Poisson 1 (lambda: {}) w/ exp({})".format(self.lambda_1, self.exp_beta_1))

        plt.scatter(self.poisson_2.timestamps, self.poisson_exp_2, c='blue',
                    label="Poisson 1 (lambda: {}) w/ exp({})".format(self.lambda_2, self.exp_beta_2))

        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("y = exp(mu)")
        plt.show()

