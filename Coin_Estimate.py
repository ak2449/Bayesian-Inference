import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the prior
def uniform_prior(theta):
    """
    Uniform prior: All values of theta (coin bias) are equally likely.
    """
    return 1 if 0 <= theta <= 1 else 0

# Step 2: Define the likelihood
def likelihood(theta, data):
    """
    Likelihood function: P(Data | Theta)
    Binomial likelihood: P(Data | Theta) = Theta^H * (1 - Theta)^T
    where H = number of heads, T = number of tails
    """
    heads = sum(data)
    tails = len(data) - heads
    return (theta ** heads) * ((1 - theta) ** tails)

# Step 3: Define the posterior (unnormalized)
def posterior(theta, data):
    """
    Posterior (unnormalized): P(Theta | Data) = P(Data | Theta) * P(Theta)
    """
    return likelihood(theta, data) * uniform_prior(theta)

# Step 4: Generate data
# Observed coin flips: 1 = heads, 0 = tails
data = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1]  # 7 heads, 3 tails

# Step 5: Compute posterior for a range of theta values
theta_values = np.linspace(0, 1, 1000)  # Range of possible coin biases
posterior_values = np.array([posterior(theta, data) for theta in theta_values])

# Normalize the posterior (to make it a valid probability distribution)
posterior_values /= posterior_values.sum()

# Step 6: Plot the posterior distribution
plt.figure(figsize=(10, 6))
plt.plot(theta_values, posterior_values, label="Posterior", color="blue")
plt.title("Posterior Distribution of Coin Bias (Theta)")
plt.xlabel("Theta (Coin Bias)")
plt.ylabel("Probability Density")
plt.legend()
plt.grid()
plt.show()