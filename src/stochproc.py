import numpy as np

# def OU_proc_inexact(x_0, k, theta, sigma, dt, n_steps):
#     """
#     Simulate Ornstein-Uhlenbeck process
#     """
#     x = np.zeros(n_steps)
#     x[0] = x_0

#     for i in range(1, n_steps):
#         x[i] = x[i-1] + k*(theta - x[i-1])*dt + sigma*np.sqrt(dt)*np.random.normal(0, 1)
#     return x

def ou(x0, kappa_fun, mean_fun, sigma_fun, dt, n_steps):
    """
    Simulate Ornstein-Uhlenbeck process
    """
    x = np.empty(n_steps)
    x[0] = x0

    for i in range(1, n_steps):
        k = kappa_fun(i)
        theta = mean_fun(i)
        sigma = sigma_fun(i)
        x[i] = x[i-1]*np.exp(-k*dt) + theta*(1 - np.exp(-k*dt)) + sigma*np.sqrt((1-np.exp(-2*k*dt))/(2*k))*np.random.normal(0, 1)
    return x

def ols_to_ou_params(alpha, beta, dt):
    kappa = -np.log(beta)/dt
    theta = alpha/(1-np.exp(-kappa*dt))
    return kappa, theta    