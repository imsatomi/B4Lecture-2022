import argparse
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

import kmeans


def init_random(data, num_c):
    """define initial clusters randomly
    Args:
        data (ndarray): imput data
        num_c (int): number of clusters
    Returns:
        clusters (ndarray)
    """

    pi = np.full(num_c, 1 / num_c)
    mu = np.random.randn(num_c, data.shape[1])
    sigma = np.array([np.eye(data.shape[1]) for i in range(num_c)])

    return pi, mu, sigma


def gaussian(data, mu, sigma):
    """calculate multi-dimensional Gaussian distribution of one data point
    Args:
        data (ndarray):input data
        mu (ndarray):centroid
        sigma (ndarray):covariance matrix
    Returns:
        nume / deno (ndarray):multi-dimensional Gaussian distribution
    """

    D = data.shape[1]
    K = mu.shape[0]  # num_c
    diff_data = np.array([data - mu[i] for i in range(K)])

    inexp = diff_data @ np.linalg.inv(sigma) @ diff_data.transpose(0, 2, 1)
    inexp = np.diagonal(inexp, axis1=1, axis2=2)

    # numerator
    nume = np.exp(-inexp / 2)
    # denominator
    deno = np.sqrt(((2 * np.pi) ** D) * np.linalg.det(sigma)).reshape(-1, 1)

    return nume / deno


def mix_gaussian(data, pi, mu, sigma):
    """calculate GMM
    Args:
        data (ndarray):input data
        pi (ndarray):mixing coefficient
        mu (ndarray):centroid
        sigma (ndarray):covariance matrix
    Returns:
        m_gauss (ndarray):probability density of GMM
        w_gauss (ndarray):gaussian * pi
    """

    w_gauss = gaussian(data, mu, sigma) * pi.reshape(-1, 1)
    m_gauss = np.sum(w_gauss, axis=0)

    return m_gauss, w_gauss


def log_likelihood(m_gaussian, log=True):
    """calculate log-likelihood
    Args:
        m_gaussian (ndarray):probability density of GMM
    Returns:
        likelihood (float):log-likelihood
    """

    if log:
        likelihood = np.sum(np.log(m_gaussian))
    else:
        likelihood = np.prod(m_gaussian)

    return likelihood


def em_algorithm(data, pi, mu, sigma, epsilon=1e-3):
    """EM algorithm
    Args:
        data (ndarray):input data
        pi (ndarray):mixing coefficient
        mu (ndarray):centroid
        sigma (ndarray):covariance matrix
        epsolon (float):convergence condition for log-likelihood
    Returns:
        likelihoods (list of float):log-likelihood
        pi (ndarray):mixing coefficient
        mu (ndarray):centroid
        sigma (ndarray):covariance matrix
    """

    N, D = data.shape
    K = mu.shape[0]  # num_c

    m_gauss, w_gauss = mix_gaussian(data, pi, mu, sigma)
    pre_likelihood = -np.inf
    likelihood = log_likelihood(m_gauss)
    likelihoods = [likelihood]
    count = 0
    max_iter = 300

    while count < max_iter:
        # E step
        burden_rate = w_gauss / m_gauss

        # M step
        N_k = burden_rate.sum(axis=1)
        pi = N_k / N

        mu = np.sum(
            burden_rate.reshape(burden_rate.shape[0], burden_rate.shape[1], 1)
            * data
            / N_k.reshape(-1, 1, 1),
            axis=1,
        )
        diff_data = np.array([data - mu[i] for i in range(K)])
        sigma = (
            burden_rate.reshape(burden_rate.shape[0], 1, burden_rate.shape[1])
            * diff_data.transpose(0, 2, 1)
            @ diff_data
        ) / N_k.reshape(-1, 1, 1)
        m_gauss, w_gauss = mix_gaussian(data, pi, mu, sigma)

        # likelihood
        pre_likelihood = likelihood
        likelihood = log_likelihood(m_gauss)
        likelihoods.append(likelihood)
        count += 1

        if likelihood - pre_likelihood < epsilon:
            print("count:", count)

            return likelihoods, pi, mu, sigma

    return likelihoods, pi, mu, sigma


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("filepath", type=str, help="csv file name")
    parser.add_argument("c", type=int, help="number of clusters")
    parser.add_argument("--init", choices=["random", "kmeans", "kmeans++"])
    args = parser.parse_args()

    filepath = args.filepath
    filename = filepath.split(".")[0]
    num_c = args.c

    data = pd.read_csv(f"../data/{filepath}", header=None).values
    # data = pd.read_csv(f'data/{filepath}', header = None).values
    # print('data.shape', data.shape)
    start = time.time()

    if args.init == "random":
        print("random")
        pi, mu, sigma = init_random(data, num_c)

    elif args.init == "kmeans":
        print("kmeans")
        centroid = kmeans.init_random(data, num_c)
        cluster, centroid = kmeans.k_means(data, num_c, centroid)
        cov = np.zeros((num_c, 2, 2))
        for i in range(num_c):
            cov[i] = np.cov(data[np.where(cluster == i)].T)
        pi = np.full(num_c, 1 / num_c)
        mu = centroid
        sigma = cov

    elif args.init == "kmeans++":
        print("kmeans++")
        centroid = kmeans.k_means_plusplus(data, num_c)
        cluster, centroid = kmeans.k_means(data, num_c, centroid)
        cov = np.zeros((num_c, 2, 2))
        for i in range(num_c):
            cov[i] = np.cov(data[np.where(cluster == i)].T)
        pi = np.full(num_c, 1 / num_c)
        mu = centroid
        sigma = cov

    likelihoods, pi, mu, sigma = em_algorithm(data, pi, mu, sigma)

    # time
    calc_time = time.time() - start
    print(f"calc_time({args.init}): {calc_time:.5f}")

    if data.shape[1] == 1:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set(
            xlabel="x",
            ylabel="probability density",
            title=f"{filename} K={num_c} {args.init}",
        )

        pos = np.linspace(np.min(data) - 1, np.max(data) + 1, 100)
        z = np.zeros(100)
        for k in range(num_c):
            z += pi[k] * multivariate_normal.pdf(pos, mu[k], sigma[k])

        ax.scatter(
            data,
            np.zeros_like(data),
            edgecolors="blueviolet",
            facecolor="None",
            label="Observed data",
        )
        plt.plot(pos, z, c="orange", label="GMM")
        ax.scatter(mu, np.zeros(num_c), marker="*", s=100, c="k", label="Centroid")
        plt.legend()
        plt.tight_layout()
        plt.savefig("fig/" + f"{filename}_k{num_c}_{args.init}.png")
        plt.show()
        plt.close()

        # plot likelihood
        fig, ax = plt.subplots()
        ax.set(
            xlabel="Iteration", ylabel="Log Likelihood", title=f"{filename} likelihood"
        )

        plt.plot(np.arange(0, len(likelihoods), 1), likelihoods)
        plt.savefig("fig/" + f"{filename}_k{num_c}_{args.init}lh.png")
        plt.show()

    if data.shape[1] == 2:

        x1 = np.linspace(np.min(data[:, 0]) - 1, np.max(data[:, 0]) + 1, 100)
        x2 = np.linspace(np.min(data[:, 1]) - 1, np.max(data[:, 1]) + 1, 100)
        x1, x2 = np.meshgrid(x1, x2)
        pos = np.dstack((x1, x2))

        prob = np.array([mix_gaussian(x_pos, pi, mu, sigma)[0] for x_pos in pos])

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set(
            xlabel="$x_1$", ylabel="$x_2$", title=f"{filename} K={num_c} {args.init}"
        )

        ax.scatter(
            data[:, 0],
            data[:, 1],
            edgecolors="blueviolet",
            facecolor="None",
            label="Observed data",
        )
        ax.scatter(mu[:, 0], mu[:, 1], marker="*", s=100, c="k", label="Centroid")
        plt.contour(x1, x2, prob, cmap="rainbow")

        plt.legend()
        plt.tight_layout()
        plt.savefig("fig/" + f"{filename}_k{num_c}_{args.init}.png")
        plt.show()
        plt.close()

        # plot likelihood
        fig, ax = plt.subplots()
        ax.set(
            xlabel="Iteration", ylabel="Log Likelihood", title=f"{filename} likelihood"
        )

        plt.plot(np.arange(0, len(likelihoods), 1), likelihoods)
        plt.savefig("fig/" + f"{filename}_k{num_c}_{args.init}lh.png")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
