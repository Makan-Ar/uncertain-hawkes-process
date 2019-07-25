import numpy as np
from scipy.stats import expon


def hawkes_log_likelihood_numpy(event_times, intensity, alpha, beta):
    """
    Returns Hawkes log likelihood.

    :param event_times: (float) list of all event times
    :param intensity: (float) intensity of the Hawkes process
    :param alpha: (float) alpha of the Hawkes process
    :param beta: (float) beta of the Hawkes process
    """
    a_calc = np.zeros(len(event_times))
    for i in range(1, len(event_times)):
        a_calc[i] = np.exp(-1 * beta * (event_times[i] - event_times[i - 1])) * (1 + a_calc[i - 1])

    term1 = np.sum(np.log(intensity + alpha * a_calc))

    term2 = intensity * event_times[-1]

    ker_ = np.sum(np.exp(-1 * beta * (event_times[-1] - event_times))) - len(event_times)
    term3 = (alpha / beta) * ker_

    res = term1 - term2 + term3
    return res


def poisson_log_likelihood_numpy(num_events, intensity, end_time):
    # TODO: Make sure this is the correct formula, and if so, what end time should be used.
    """
    Returns Poisson process log likelihood.

    Based on https://stats.stackexchange.com/questions/360814/mle-for-a-homogeneous-poisson-process and
    https://math.stackexchange.com/questions/344487/log-likelihood-of-a-realization-of-a-poisson-process

    :param num_events: (int) number of event times.
    :param intensity: (float) rate/intensity of the Poisson process.
    :param end_time: (float) end time of the current sequence.
    :return:
    """
    return num_events * np.log(intensity) - intensity * end_time


def z_i_posterior_prob(event_time, event_mark, event_times_hist, z_hist, z_prior,
                       hawkes_params, poisson_lambda,
                       hawkes_mark_exp_rate, noise_mark_exp_rate):
    """
    Returns p(z_i=1 | T_1:i, y_i, Z_1:i-1)

    :param event_time: (float) time of the new event i
    :param event_mark: (float) mark of the new event i
    :param event_times_hist: list of event times from t_0 to t_i-1
    :param z_hist: list of booleans to identify event_times_hist events as Hawkes (0/false) or noise/poisson (True)
    :param z_prior: prior probability of latent variable Z=1 (prior probability of noise)
    :param hawkes_params: a tuple of hawkes parameters (lambda, alpha, beta)
    :param poisson_lambda: lambda parameter of the poisson process (noise)
    :param hawkes_mark_exp_rate: lambda parameter of the exponential dist for the marks of the Hawkes process events.
    :param noise_mark_exp_rate: lambda parameter of the exponential dist for the marks of the noise events.
    """
    # noise/hawkes at the end of each variable indicates whether z_i was assumed to be 1 or 0.

    mark_prob_noise = expon.pdf(event_mark, scale=1./noise_mark_exp_rate)
    mark_prob_hawkes = expon.pdf(event_mark, scale=1./hawkes_mark_exp_rate)

    hawkes_intensity, hawkes_alpha, hawkes_beta = hawkes_params
    hawkes_prob_noise = np.exp(hawkes_log_likelihood_numpy(event_times_hist[np.logical_not(z_hist)],
                                                           hawkes_intensity, hawkes_alpha, hawkes_beta))

    hawkes_prob_hawkes = np.exp(hawkes_log_likelihood_numpy(
        np.append(event_times_hist[np.logical_not(z_hist)], event_time), hawkes_intensity, hawkes_alpha, hawkes_beta))

    poisson_prob_noise = np.exp(poisson_log_likelihood_numpy(np.sum(z_hist) + 1, poisson_lambda, event_times_hist[-1]))
    poisson_prob_hawkes = np.exp(poisson_log_likelihood_numpy(np.sum(z_hist), poisson_lambda, event_times_hist[-1]))

    numerator = z_prior * mark_prob_noise * hawkes_prob_noise * poisson_prob_noise
    normalizer = ((1 - z_prior) * mark_prob_hawkes * hawkes_prob_hawkes * poisson_prob_hawkes) + numerator

    return numerator / normalizer


def z_i_posterior_log_prob(event_time, event_mark, event_times_hist, z_hist, z_prior,
                           hawkes_params, poisson_lambda,
                           hawkes_mark_exp_rate, noise_mark_exp_rate):
    """
    Returns ln p(z_i=1 | T_1:i, y_i, Z_1:i-1)

    Check out z_i_posterior_prob doc.
    """
    return np.log(z_i_posterior_prob(event_time, event_mark, event_times_hist, z_hist, z_prior,
                  hawkes_params, poisson_lambda,
                  hawkes_mark_exp_rate, noise_mark_exp_rate))


def z_posterior_prob(z, event_times, event_marks, z_prior,
                     hawkes_params, poisson_lambda,
                     hawkes_mark_exp_rate, noise_mark_exp_rate):
    # TODO: Take care of i=1 prob.
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
    z_i_probs = np.ones(len(event_times))
    for i in range(1, len(event_times)):
        z_i_probs[i] = z_i_posterior_prob(event_times[i], event_marks[i], event_times[:i], z[:i], z_prior,
                                          hawkes_params, poisson_lambda, hawkes_mark_exp_rate, noise_mark_exp_rate)

    return np.prod(z_i_probs)


def z_posterior_log_prob(z, event_times, event_marks, z_prior,
                         hawkes_params, poisson_lambda,
                         hawkes_mark_exp_rate, noise_mark_exp_rate):
    # TODO: Take care of i=1 prob.
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
    for i in range(1, len(event_times)):
        z_i_probs[i] = z_i_posterior_log_prob(event_times[i], event_marks[i], event_times[:i], z[:i], z_prior,
                                              hawkes_params, poisson_lambda, hawkes_mark_exp_rate, noise_mark_exp_rate)

    return np.sum(z_i_probs)