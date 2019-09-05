import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# A-B test
p = pm.Uniform('p', lower=0, upper=1)
p_true = 0.05 # Unknown
N = 1500

# Ber(0.05) simmulation
occur = pm.rbernoulli(p_true, N)
print(occur)
print(occur.sum())
print(occur.mean())

obs = pm.Bernoulli("obs", p, value=occur, observed=True)
mcmc = pm.MCMC([p, obs])
mcmc.sample(20000, 1000)

plt.figure(figsize=(12.5, 4))
plt.vlines(p_true, 0, 90, linestyle='--', label="real $p_A$ unknown value")
plt.hist(mcmc.trace("p")[:], bins=25, histtype="stepfilled", normed=True)
plt.legend()
plt.show()

true_p_a = 0.05
true_p_b = 0.04
n_a = 1500
n_b = 750

obs_a = pm.rbernoulli(true_p_a, n_a)
obs_b = pm.rbernoulli(true_p_b, n_b)

p_a = pm.Uniform('p_a', 0, 1)
p_b = pm.Uniform('p_b', 0, 1)

@pm.deterministic
def delta(p_a=p_a, p_b=p_b):
    return p_a - p_b

obs_a = pm.Bernoulli("obs_a", p_a, value=obs_a, observed=True)
obs_b = pm.Bernoulli("obs_b", p_b, value=obs_b, observed=True)

mcmc = pm.MCMC([p_a, p_b, delta, obs_a, obs_b])
mcmc.sample(25000, 5000)

p_a_samples = mcmc.trace("p_a")[:]
p_b_samples = mcmc.trace("p_b")[:]
delta_samples = mcmc.trace("delta")[:]

ax = plt.subplot(311)
plt.xlim(0, .1)
plt.hist(p_a_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="$p_A$ posterior pd", color='#A60628', normed=True)
plt.vlines(true_p_a, 0, 80, linestyle='--', label="$p_A$ (unknown)")
plt.legend(loc="upper right")
plt.ylim(0, 80)

ax = plt.subplot(312)
plt.xlim(0, .1)
plt.hist(p_b_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="$p_b$ posterior pd", color='#467821', normed=True)
plt.vlines(true_p_b, 0, 80, linestyle='--', label="$p_b$ (unknown)")
plt.legend(loc="upper right")
plt.ylim(0, 80)

ax = plt.subplot(313)
plt.hist(delta_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="delta posterior pd", color='#7A68A6', normed=True)
plt.vlines(true_p_b-true_p_b, 0, 60, linestyle='--', label="delta (unknown)")
plt.vlines(0, 0, 60, color='black', alpha=0.2)
plt.legend(loc="upper right")

## 거짓말 알고리즘 --> 이항분포
binomial = stats.binom
parameters = [(10, .4), (10, .9)]
colors = ["#348ABD", "#A60628"]

for i in range(2):
    N, p = parameters[i]
    _x = np.arange(N + 1)
    plt.bar(_x - 0.5, binomial.pmf(_x, N, p), color=colors[i], edgecolor=colors[i], alpha=0.6,
            label="$N$: %d, $p$: %.1f" %(N, p), linewidth=3)
    
plt.legend(loc="upper left")
plt.xlim(0, 10.5)
plt.xlabel("$k$")
plt.ylabel("$P(X = k)$")


















