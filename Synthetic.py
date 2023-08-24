# imports
from scipy.optimize import brentq
from scipy.stats import skew
from math import sqrt
import numpy as np


class Synthetic:

    __slots__ = ('y', 'd', 'mu_t', 'sigma_t', 'Y_t')

    def __init__(self, y, d_mu, d_sigma, d_skew):
        self.y = y

        # calculate original distribution statistics
        sigma_y = sqrt(np.var(y))
        mu_y = np.mean(y)
        Y_y = skew(y)

        # calculate desired statistics
        mu_t = mu_y + mu_y * d_mu
        sigma_t = sigma_y + sigma_y * d_sigma
        Y_t = Y_y + abs(Y_y) * d_skew

        self.mu_t = mu_t
        self.sigma_t = sigma_t
        self.Y_t = Y_t
        self.d = brentq(f=self.func, a=-10, b=50, args=(Y_t, y))

    def gen(self):
        # create intermediate distributions for adjusting skew, sigma, and mu
        g = self.sigma_t / sqrt(np.var(self.y + self.d * self.y ** 2))
        G = g * (self.y + self.d * self.y ** 2)
        h = self.mu_t - np.mean(G)

        # return synthetic distribution
        return (g * self.d * self.y ** 2 + g * self.y + h).flatten()

    def func(self, d, Y_t, y):
        n = len(y)
        num = (1 / n) * np.sum(((y + d * y ** 2) - ((1 / n) * np.sum(y + d * y ** 2))) ** 3)
        denom = (1 / (n - 1) * np.sum(((y + d * y ** 2) - ((1 / n) * np.sum(y + d * y ** 2)) ** 2) ** 2)) ** (3 / 2)
        return num / denom - Y_t
