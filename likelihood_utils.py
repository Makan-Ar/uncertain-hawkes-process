import sys
import numpy as np
from scipy.stats import expon, bernoulli, binom
from hawkes_uncertain_simulator import HawkesUncertainModel


def hawkes_log_likelihood_numpy(event_times, intensity, alpha, beta, end_time=None):
    """
    Returns Hawkes log likelihood.

    :param event_times: (float) list of all event times
    :param intensity: (float) intensity of the Hawkes process
    :param alpha: (float) alpha of the Hawkes process
    :param beta: (float) beta of the Hawkes process
    :param end_time: optional (float) end time of the observed interval. Default is the last event time.
    """

    if end_time is None and len(event_times) == 0:
        sys.exit("Error evaluating Hawkes log-likelihood: end_time must be defined if there is no observed event.")

    if end_time is not None and len(event_times) > 0 and end_time < event_times[-1]:
        sys.exit("Error evaluating Hawkes log-likelihood: end_time must be >= to the last event time.")

    end_time = end_time if end_time is not None else event_times[-1]
    num_events = len(event_times)

    term1 = 0
    if num_events > 0:
        a_calc = np.zeros(num_events)
        for i in range(1, num_events):
            a_calc[i] = np.exp(-1 * beta * (event_times[i] - event_times[i - 1])) * (1 + a_calc[i - 1])

        term1 = np.sum(np.log(intensity + alpha * a_calc))

    term2 = intensity * end_time
    term3 = (alpha / beta) * np.sum(np.exp(-1 * beta * (end_time - event_times))) - num_events

    res = term1 - term2 + term3
    return res


def poisson_log_likelihood_numpy(num_events, intensity, end_time):
    """
    Returns Poisson process log likelihood.

    Based on https://stats.stackexchange.com/questions/360814/mle-for-a-homogeneous-poisson-process and
    https://math.stackexchange.com/questions/344487/log-likelihood-of-a-realization-of-a-poisson-process

    :param num_events: (int) number of event times.
    :param intensity: (float) rate/intensity of the Poisson process.
    :param end_time: (float) end time of the observed interval. If the last poisson event time < end of the observed
                             interval, end_time must be set to end of the observed interval.
    :return:
    """
    return num_events * np.log(intensity) - intensity * end_time


def z_i_posterior_prob(event_time, event_mark, event_times_hist, z_hist, z_prior,
                       hawkes_params, poisson_lambda,
                       hawkes_mark_exp_rate, noise_mark_exp_rate, return_prob_z_i_noise=True):
    """
    Returns p(z_i=1 | T_1:i, y_i, Z_1:i-1) if `return_prob_z_i_noise`=True, else p(z_i=0 | T_1:i, y_i, Z_1:i-1)

    :param event_time: (float) time of the new event i
    :param event_mark: (float) mark of the new event i
    :param event_times_hist: list of event times from t_0 to t_i-1
    :param z_hist: list of booleans to identify event_times_hist events as Hawkes (0/false) or noise/poisson (True)
    :param z_prior: prior probability of latent variable Z=1 (prior probability of noise)
    :param hawkes_params: a tuple of hawkes parameters (lambda, alpha, beta)
    :param poisson_lambda: lambda parameter of the poisson process (noise)
    :param hawkes_mark_exp_rate: lambda parameter of the exponential dist for the marks of the Hawkes process events.
    :param noise_mark_exp_rate: lambda parameter of the exponential dist for the marks of the noise events.
    :param return_prob_z_i_noise: optional (bool), if true returns the probability of z_i = 1, and z_i = 0 otherwise
    """
    # noise/hawkes at the end of each variable indicates whether z_i was assumed to be 1 or 0.
    z_not_hist = np.logical_not(z_hist)

    mark_prob_noise = expon.pdf(event_mark, scale=1./noise_mark_exp_rate)
    mark_prob_hawkes = expon.pdf(event_mark, scale=1./hawkes_mark_exp_rate)

    hawkes_intensity, hawkes_alpha, hawkes_beta = hawkes_params
    hawkes_prob_noise = np.exp(hawkes_log_likelihood_numpy(event_times_hist[z_not_hist],
                                                           hawkes_intensity, hawkes_alpha, hawkes_beta, event_time))

    hawkes_prob_hawkes = np.exp(hawkes_log_likelihood_numpy(np.append(event_times_hist[z_not_hist], event_time),
                                                            hawkes_intensity, hawkes_alpha, hawkes_beta))

    poisson_prob_noise = np.exp(poisson_log_likelihood_numpy(np.sum(z_hist) + 1, poisson_lambda, event_time))
    poisson_prob_hawkes = np.exp(poisson_log_likelihood_numpy(np.sum(z_hist), poisson_lambda, event_time))

    numerator = z_prior * mark_prob_noise * hawkes_prob_noise * poisson_prob_noise
    normalizer = ((1 - z_prior) * mark_prob_hawkes * hawkes_prob_hawkes * poisson_prob_hawkes) + numerator

    z_i_noise_prob = numerator / normalizer
    if return_prob_z_i_noise:
        return z_i_noise_prob

    return 1 - z_i_noise_prob


def z_i_posterior_log_prob(event_time, event_mark, event_times_hist, z_hist, z_prior,
                           hawkes_params, poisson_lambda,
                           hawkes_mark_exp_rate, noise_mark_exp_rate, return_prob_z_i_noise=True):
    """
    Returns ln p(z_i=1 | T_1:i, y_i, Z_1:i-1) if `return_prob_z_i_noise`=True, else ln p(z_i=0 | T_1:i, y_i, Z_1:i-1)

    Check out z_i_posterior_prob doc.
    """
    return np.log(z_i_posterior_prob(event_time, event_mark, event_times_hist, z_hist, z_prior,
                  hawkes_params, poisson_lambda,
                  hawkes_mark_exp_rate, noise_mark_exp_rate, return_prob_z_i_noise))


def z_posterior_prob(z, event_times, event_marks, z_prior,
                     hawkes_params, poisson_lambda,
                     hawkes_mark_exp_rate, noise_mark_exp_rate):
    """
    Returns product of p(z_i=1 | T_1:i, y_i, Z_1:i-1) for i from 0 to len(event_times)

    :param z: list of boolean. The list of all z_i's. True is noise, False is Hawkes.
    :param event_times: list of all event times
    :param event_marks: list of all event marks
    :param z_prior: prior probability of latent variable Z=1 (prior probability of noise)
    :param hawkes_params: a tuple of hawkes parameters (lambda, alpha, beta)
    :param poisson_lambda: lambda parameter of the poisson process (noise)
    :param hawkes_mark_exp_rate: lambda parameter of the exponential dist for the marks of the Hawkes process events.
    :param noise_mark_exp_rate: lambda parameter of the exponential dist for the marks of the noise events.
    """
    z_i_probs = np.zeros(len(event_times))
    for i in range(0, len(event_times)):
        z_i_probs[i] = z_i_posterior_prob(event_times[i], event_marks[i], event_times[:i], z[:i], z_prior,
                                          hawkes_params, poisson_lambda, hawkes_mark_exp_rate, noise_mark_exp_rate,
                                          return_prob_z_i_noise=z[i] == 1)
        print(i, event_times[i], z[i], z_i_probs[i])
    return np.prod(z_i_probs)


def z_posterior_log_prob(z, event_times, event_marks, z_prior,
                         hawkes_params, poisson_lambda,
                         hawkes_mark_exp_rate, noise_mark_exp_rate):
    """
    Returns sum of ln p(z_i=1 | T_1:i, y_i, Z_1:i-1) for i from 0 to len(event_times)

    :param z: list of boolean. The list of all z_i's. True is noise, False is Hawkes.
    :param event_times: list of all event times
    :param event_marks: list of all event marks
    :param z_prior: prior probability of latent variable Z=1 (prior probability of noise)
    :param hawkes_params: a tuple of hawkes parameters (lambda, alpha, beta)
    :param poisson_lambda: lambda parameter of the poisson process (noise)
    :param hawkes_mark_exp_rate: lambda parameter of the exponential dist for the marks of the Hawkes process events.
    :param noise_mark_exp_rate: lambda parameter of the exponential dist for the marks of the noise events.
    """
    z_i_probs = np.zeros(len(event_times))
    for i in range(0, len(event_times)):
        z_i_probs[i] = z_i_posterior_log_prob(event_times[i], event_marks[i], event_times[:i], z[:i], z_prior,
                                              hawkes_params, poisson_lambda, hawkes_mark_exp_rate, noise_mark_exp_rate,
                                              return_prob_z_i_noise=z[i] == 1)

    return np.sum(z_i_probs)


def z_posterior_log_prob_type_2(z, event_times, event_marks, z_prior,
                                hawkes_params, poisson_lambda,
                                hawkes_mark_exp_rate, noise_mark_exp_rate):
    """
    Returns log prob as the sum of each individual log prob. Not normalized.

    ln p(Z/bernoulli) + ln p(mark/exp) + ln prob hawkes + ln prob poisson. This is not normalized, since the denominator is
    intractable.

    :param z: list of boolean. The list of all z_i's. True is noise, False is Hawkes.
    :param event_times: list of all event times
    :param event_marks: list of all event marks
    :param z_prior: prior probability of latent variable Z=1 (prior probability of noise)
    :param hawkes_params: a tuple of hawkes parameters (lambda, alpha, beta)
    :param poisson_lambda: lambda parameter of the poisson process (noise)
    :param hawkes_mark_exp_rate: lambda parameter of the exponential dist for the marks of the Hawkes process events.
    :param noise_mark_exp_rate: lambda parameter of the exponential dist for the marks of the noise events.
    """
    z_poisson = z
    z_hawkes = np.logical_not(z)

    # z_log_prob = np.sum(bernoulli.logpmf(z, p=z_prior))
    z_log_prob = binom.logpmf(np.sum(z_poisson), len(z_poisson), z_prior)

    mark_hawkes_log_prob = np.sum(expon.logpdf(event_marks[z_hawkes], scale=1./hawkes_mark_exp_rate))
    mark_noise_log_prob = np.sum(expon.logpdf(event_marks[z_poisson], scale=1./noise_mark_exp_rate))

    hawkes_log_prob = hawkes_log_likelihood_numpy(event_times[z_hawkes],
                                                  hawkes_params[0], hawkes_params[1], hawkes_params[2],
                                                  event_times[-1])
    poisson_log_prob = poisson_log_likelihood_numpy(np.sum(z_poisson), poisson_lambda, event_times[-1])

    return (z_log_prob +
            mark_hawkes_log_prob +
            mark_noise_log_prob +
            hawkes_log_prob +
            poisson_log_prob)


if __name__ == "__main__":
    _h_intensity = 0.9
    _h_beta = 2
    _h_alpha = 1.2

    _runtime = 50

    _p_intensity = 0.1

    _h_exp_rate = 1.5
    _p_exp_rate = 1.5

    hum = HawkesUncertainModel(h_lambda=_h_intensity, h_alpha=_h_alpha, h_beta=_h_beta, h_exp_rate=_h_exp_rate,
                               p_lambda=_p_intensity, p_exp_rate=_p_exp_rate,
                               noise_percentage_ub=0.5, run_time=_runtime, delta=0.01, seed=None)

    # Testing out the prob of the full posterior
    sim_event_times = hum.mixed_timestamps
    sim_event_marks = hum.mixed_expo
    sim_true_labels = hum.mixed_labels.astype(np.bool)
    sim_true_z_prior = hum.noise_percentage
    print(hum.noise_percentage)
    hum.plot_hawkes_uncertain()

    print("true labels:", z_posterior_log_prob(sim_true_labels,
                                               sim_event_times, sim_event_marks, sim_true_z_prior,
                                               (_h_intensity, _h_alpha, _h_beta),
                                               _p_intensity, _h_exp_rate, _p_exp_rate))

    print("all hawkes", z_posterior_log_prob(np.zeros(len(sim_true_labels)).astype(np.bool),
                                             sim_event_times, sim_event_marks, sim_true_z_prior,
                                             (_h_intensity, _h_alpha, _h_beta),
                                             _p_intensity, _h_exp_rate, _p_exp_rate))

    print("all noise", z_posterior_log_prob(np.ones(len(sim_true_labels)).astype(np.bool),
                                            sim_event_times, sim_event_marks, sim_true_z_prior,
                                            (_h_intensity, _h_alpha, _h_beta),
                                            _p_intensity, _h_exp_rate, _p_exp_rate))

    print("exact opposite", z_posterior_log_prob(np.logical_not(sim_true_labels),
                                                 sim_event_times, sim_event_marks, sim_true_z_prior,
                                                 (_h_intensity, _h_alpha, _h_beta),
                                                 _p_intensity, _h_exp_rate, _p_exp_rate))

    print("Type 2")
    print("true labels:", z_posterior_log_prob_type_2(sim_true_labels,
                                                      sim_event_times, sim_event_marks, sim_true_z_prior,
                                                      (_h_intensity, _h_alpha, _h_beta),
                                                      _p_intensity, _h_exp_rate, _p_exp_rate))

    print("all hawkes", z_posterior_log_prob_type_2(np.zeros(len(sim_true_labels)).astype(np.bool),
                                                    sim_event_times, sim_event_marks, sim_true_z_prior,
                                                    (_h_intensity, _h_alpha, _h_beta),
                                                    _p_intensity, _h_exp_rate, _p_exp_rate))

    print("all noise", z_posterior_log_prob_type_2(np.ones(len(sim_true_labels)).astype(np.bool),
                                                   sim_event_times, sim_event_marks, sim_true_z_prior,
                                                   (_h_intensity, _h_alpha, _h_beta),
                                                   _p_intensity, _h_exp_rate, _p_exp_rate))

    print("exact opposite", z_posterior_log_prob_type_2(np.logical_not(sim_true_labels),
                                                        sim_event_times, sim_event_marks, sim_true_z_prior,
                                                        (_h_intensity, _h_alpha, _h_beta),
                                                        _p_intensity, _h_exp_rate, _p_exp_rate))
