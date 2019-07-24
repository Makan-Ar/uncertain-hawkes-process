import numpy as np
import utils
from hawkes_uncertain_simulator import HawkesUncertainModel


_h_intensity = 0.5
_h_beta = 2
_h_alpha = 0.9

_runtime = 50

_p_intensity = 0.1

_h_exp_rate = 1.5
_p_exp_rate = 116.5

hum = HawkesUncertainModel(h_lambda=_h_intensity, h_alpha=_h_alpha, h_beta=_h_beta, h_exp_rate=_h_exp_rate,
                           p_lambda=_p_intensity, p_exp_rate=_p_exp_rate,
                           noise_percentage_ub=0.1, run_time=_runtime, delta=0.01, seed=3)

event_times = hum.mixed_timestamps
mixed_marks = hum.mixed_expo
true_labels = hum.mixed_labels.astype(np.bool)


# Per event log_likelihoods

# all events are Hawkes
print("All events are Hawkes:",
      utils.hawkes_log_likelihood_numpy(event_times, _h_intensity, _h_alpha, _h_beta) / len(event_times))

# all events are poisson
print("All events are Poisson:",
      utils.poisson_log_likelihood_numpy(event_times, _p_intensity, event_times[-1]) / len(event_times))

# based on true labels
print("Based on true labels:",
      (utils.hawkes_log_likelihood_numpy(event_times[np.logical_not(true_labels)], _h_intensity, _h_alpha, _h_beta) +
       utils.poisson_log_likelihood_numpy(event_times[true_labels], _p_intensity, event_times[-1])) / len(event_times))

# based on exact opposite of true labels
print("Based on exact opposite of true labels",
      (utils.hawkes_log_likelihood_numpy(event_times[true_labels], _h_intensity, _h_alpha, _h_beta) +
       utils.poisson_log_likelihood_numpy(event_times[np.logical_not(true_labels)], _p_intensity, event_times[-1])) /
      len(event_times))

# likelihood of Hawkes events only
print("Likelihood of Hawkes events only:",
      utils.hawkes_log_likelihood_numpy(event_times[np.logical_not(true_labels)], _h_intensity, _h_alpha, _h_beta) /
      np.sum(np.logical_not(true_labels)))

# likelihood of Poisson events only
print("Likelihood of Poisson events only:",
      utils.poisson_log_likelihood_numpy(event_times[true_labels], _p_intensity, event_times[-1]) / np.sum(true_labels))

