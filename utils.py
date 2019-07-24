import numpy as np
# from hawkes_uncertain_simulator import HawkesUncertainModel
#
# hum = HawkesUncertainModel(h_lambda=0.5, h_alpha=0.2, h_beta=5, h_exp_beta=0.7, p_lambda=0.5, p_exp_beta=0.5)
# hum.plot_hawkes()
# hum.plot_poisson()
# hum.plot_hawkes_uncertain()


def hawkes_log_likelihood_numpy(hawkes_event_times, intensity, alpha, beta):
    a_calc = np.zeros(len(hawkes_event_times))
    for i in range(1, len(hawkes_event_times)):
        a_calc[i] = np.exp(-1 * beta * (hawkes_event_times[i] - hawkes_event_times[i - 1])) * (1 + a_calc[i - 1])

    term1 = np.sum(np.log(intensity + alpha * a_calc))

    term2 = intensity * hawkes_event_times[-1]

    ker_ = np.sum(np.exp(-1 * beta * (hawkes_event_times[-1] - hawkes_event_times))) - len(hawkes_event_times)
    term3 = (alpha / beta) * ker_

    res = term1 - term2 + term3
    return res


# Based on https://stats.stackexchange.com/questions/360814/mle-for-a-homogeneous-poisson-process and
# https://math.stackexchange.com/questions/344487/log-likelihood-of-a-realization-of-a-poisson-process
def poisson_log_likelihood_numpy(poisson_event_times, intensity, end_time):
    return len(poisson_event_times) * np.log(intensity) - intensity * end_time

