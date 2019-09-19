import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)
c_data = np.genfromtxt("d:/data/challenger_data.csv", skip_header=1, usecols=[1,2], missing_values='NA', delimiter=',')
c_data = c_data[~np.isnan(c_data[:,1])]
print("TEMP, O-RING failure?")
print(c_data)

def logistic(x, beta, alpha=0):
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))

x = np.linspace(-4, 4, 100)
#plt.plot(x, logistic(x, 1))
#plt.plot(x, logistic(x, 3))
#plt.plot(x, logistic(x, -5))
plt.legend()
plt.plot(x, logistic(x, 1, 1))
plt.plot(x, logistic(x, 3, -2))
plt.plot(x, logistic(x, -5, 7))

temp = c_data[:, 0]
D = c_data[:, 1]

beta = pm.Normal("beta", 0, 0.001, value=0)
alpha = pm.Normal("alpha", 0, 0.001, value=0)

@pm.deterministic
def p(t=temp, alpha=alpha, beta=beta):
    return 1.0 / (1. + np.exp(beta*t+alpha))

p.value

observed = pm.Bernoulli("bernoulli_obs", p, value=D, observed=True)
model = pm.Model([observed, beta, alpha])

map_ = pm.MAP(model)
map_.fit()
mcmc = pm.MCMC(model)
mcmc.sample(120000, 100000, 2)

alpha_samples = mcmc.trace('alpha')[:, None]
beta_samples = mcmc.trace('beta')[:, None]

t = np.linspace(temp.min() - 5, temp.max() + 5, 50)[:, None]
p_t = logistic(t.T, beta_samples, alpha_samples)
mean_prob_t = p_t.mean(axis=0)

plt.plot(t, mean_prob_t, lw=3)
plt.plot(t, p_t[0, :], ls='--')
plt.plot(t, p_t[-2, :], ls="--")
plt.scatter(temp, D, color="k", s=50, alpha=0.5)