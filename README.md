# Hawkes Process with Uncertain Events (unsupervised learning)
The objective of this project is to detect noise (uncertain events) in a sequnce of event data, modeled by multivariate and univariate Hawkes processes.

## Model setup
- Noise is modeled using a Poisson process.
- Every event has an associated mark, which is assumed to be exponentially distributed, with different rates for Hawkes and Poisson events
- We define a generative model with univariate Hawkes processes
- Derive a distribution over latent variable z (whether an event is noise), using Bayesian graphical models


> This repository has been published for the sole purpose of providing more information on the aforementioned project.
