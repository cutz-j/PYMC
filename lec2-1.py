import pymc as pm
import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', family='Malgun Gothic')
lambda_ = pm.Exponential("poisson_param", 1)
data_generator = pm.Poisson("data_generater", lambda_)
data_plus_one = data_generator + 1

print(lambda_.children)
print(data_generator.parents)

# value
print(lambda_.value)

betas = pm.Uniform("betas", 0, 1, size=5)
betas.value

## random
ld1 = pm.Exponential("lambda_1", 1) # first 행동의 prior
ld2 = pm.Exponential("lambda_2", 1) # second 행동의 prior
tau = pm.DiscreteUniform("tau", lower=0, upper=10) # 행동 변화에 대한 prior

print("init")
print(ld1.value)
print(ld2.value)
print(tau.value)

print(ld1.random(), ld2.random(), tau.random())

print("random call")
print(ld1.value)
print(ld2.value)
print(tau.value)

n_data_points = 5
@pm.deterministic
def labmda_(tau=tau, lambda_1=ld1, lambda_2=ld2):
    out = np.zeros(n_data_points)
    out[:tau] = lambda_1
    out[tau:] = lambda_2
    return out

####################################################
#### 모델에 관측 포함 ####
figsize = (12.5, 4)
plt.figure(figsize=figsize)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
samples = [ld1.random() for i in range(20000)]
plt.hist(samples, bins=70, normed=True, histtype="stepfilled")
plt.xlim(0, 8)
plt.show()


# 고정 밸류
data = np.array([10, 25, 15, 20, 35])
obs = pm.Poisson("obs", lambda_, value=data, observed=True)
obs.value

##################
##### 모델링 #####

tau = pm.rdiscrete_uniform(0, 80)
alpha = 1./20.
lambda_1, lambda_2 = pm.rexponential(alpha, 2)
lambda_ = np.r_[lambda_1*np.ones(tau), lambda_2*np.ones(80-tau)]
data = pm.rpoisson(lambda_)
plt.bar(np.arange(80), data, color="#348ABD")
plt.bar(tau-1, data[tau-1], color='r', label='행동변화')
plt.xlable("time")
plt.ylabel("message")
plt.xlim(0, 80)
plt.legend()
        























    