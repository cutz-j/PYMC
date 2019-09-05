import pymc as pm
import matplotlib.pyplot as plt
import numpy as np

N = 100
p = pm.Uniform("freq_cheating", 0, 1)

true_answers = pm.Bernoulli("truths", p, size=N)
first_coin = pm.Bernoulli("first_flip", 0.5, size=N)
second_coin = pm.Bernoulli("second_flip", 0.5, size=N)

@pm.deterministic
def observed_proportion(t_a=true_answers, fc=first_coin, sc=second_coin):
    observed = fc * t_a + (1 - fc) * sc
    return observed.sum() / float(N)

observed_proportion.value

# data generation

X = 35
observations = pm.Binomial("obs", N, observed_proportion, observed=True, value=X)
model = pm.Model([p, true_answers, first_coin, second_coin, observed_proportion, observations])
mcmc = pm.MCMC(model)
mcmc.sample(40000, 15000)


p_trace = mcmc.trace("freq_cheating")[:]
plt.hist(p_trace, histtype="stepfilled", normed=True, alpha=0.85, bins=30, color="#348ABD")
plt.vlines([.05, .35], [0, 0], [5, 5], alpha=0.3)
plt.xlim(0, 1)
plt.legend()