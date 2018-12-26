from hawkes_uncertain_simulator import HawkesUncertainModel

hum = HawkesUncertainModel(h_lambda=0.5, h_alpha=0.2, h_beta=5, h_exp_beta=0.7, p_lambda=0.5, p_exp_beta=0.5)
hum.plot_hawkes()
hum.plot_poisson()
hum.plot_hawkes_uncertain()