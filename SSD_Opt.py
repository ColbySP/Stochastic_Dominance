# imports
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from Synthetic import Synthetic
from scipy import stats
import pandas as pd
import numpy as np
import time

plt.style.use('ggplot')


class SSD_Portfolio:
    __slots__ = ('tickers', 't', 'latest_prices', 'R', 'B_original',
                 'B', 'W', 'S', 'n', 'v', 'w', 'B_sort', 'W_sort', 'B_original_sort', 'Y')

    def __init__(self, return_dir: str, benchmark_dir: str):
        print("Loading data ...\n")
        # load returns data
        df = pd.read_csv(return_dir).set_index("Date").ffill(axis=1).pct_change(1).iloc[1:]
        self.tickers = df.columns.tolist()
        self.R = df.values
        self.latest_prices = self.R[-1, :]

        # load reference data
        df = pd.read_csv(benchmark_dir, parse_dates=['Date'], index_col='Date').pct_change(1)
        self.B_original = df.values[1:].flatten()
        self.B_original_sort = np.sort(self.B_original)
        self.t = df.index.values

        # placeholder for synthetic benchmark
        self.B = self.B_original

        # get the shape of the returns matrix
        self.S, self.n = self.R.shape

        # placeholder values for weights and objective
        self.v = None
        self.w = None

        # placeholder for portfolio returns
        self.W = None

        # placeholder for sorted returns
        self.B_sort = None
        self.W_sort = None

        # placeholder for cdf y_values
        self.Y = np.linspace(0, 1, num=self.S, endpoint=True)

    def adjust_benchmark(self, delta_mu: float, delta_sigma: float, delta_skew: float) -> np.array:
        """Function to generate a synthetic benchmark based on the original as outlined by Valle et al. 2017"""
        self.B = Synthetic(self.B_original, delta_mu, delta_sigma, delta_skew).gen()

    def optimize(self, time_limit: float = 60, iter_limit: int = 500) -> None:
        """Function using linear programming approach outlined by Fabian et al. 2011b to solve for optimal weights"""
        # sort benchmark returns and save indices
        r_b_i = np.argsort(self.B)
        self.B_sort = self.B[r_b_i]

        # compute tau tail values
        tau = np.cumsum(self.B_sort) / self.S

        # create the objective function to maximize v
        c = np.zeros(self.n + 1)
        c[0] = 1

        # define A_ub as an arbitrary cut of J_0
        A_ub = np.ones(shape=(1, self.n + 1))
        A_ub[:, 1:] = -self.R[0, :] / self.S

        # create b_ub as tau_0
        b_ub = np.full(shape=(1, 1), fill_value=-tau[0])

        # define a_eq to ensure sum of weights adds to a constant (don't include v)
        A_eq = np.ones(shape=(1, self.n + 1))
        A_eq[:, 0] = 0

        # define b_eq to set constant for the sum of weights to one
        b_eq = np.ones(shape=(1, 1))

        # define unbounded limits for v and bounds for each asset weight
        bounds = [(None, None)]
        for i in range(self.n):
            bounds.append((0, 1))

        # run iterative portfolio optimization
        print('Computing optimal portfolio ...\n')
        start_time = time.time()
        iter_num = 0
        while True:

            # limit computation time
            if (time.time() - start_time) > time_limit:
                print(f'Time limit reached after {iter_num} iterations!\n')
                break
            if iter_num > iter_limit:
                print(f'Iteration limit reached after {time.time() - start_time:.1f} seconds!\n')
                break

            # run optimization
            res = linprog(c=-c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

            # ensure valid solution
            if res.x is not None:
                self.v, self.w = res.x[0], res.x[1:]

            # otherwise end iteration process
            else:
                print('Infeasible problem constraints ... returning closest solution!\n')
                break

            # construct the set J^*
            self.W = np.dot(self.R, self.w)
            J_star = np.argsort(self.W)
            self.W_sort = self.W[J_star]

            # compute the residuals for each tail_i
            scaled_v = np.linspace(self.v / self.S, self.v, self.S)
            port_tau = np.cumsum(np.dot(self.R[J_star, :], self.w)) / self.S
            residuals = scaled_v + tau - port_tau

            # find the worst violation
            i_hat = np.argmax(residuals)

            if residuals[i_hat] <= 1e-7:  # set a tolerance
                print(f'Optimal portfolio achieved in {iter_num} iterations! ({time.time() - start_time:.1f}s)\n')
                break

            # enforce worst offending tail constraint
            # create new A_ub row
            new_A_ub_row = np.ones(shape=(1, self.n + 1))
            new_A_ub_row[:, 1:] = np.sum(-self.R[J_star[:i_hat + 1], :], axis=0) / self.S

            # create new b_ub row
            new_b_ub_row = -tau[i_hat]

            # create new A_eq row
            new_A_eq_row = np.ones(shape=(1, self.n + 1))
            new_A_eq_row[:, 0] = 0

            # create new b_eq row
            new_b_eq_row = 1

            # append them to current constraints
            A_ub = np.vstack((A_ub, new_A_ub_row))
            b_ub = np.append(b_ub, new_b_ub_row)
            A_eq = np.vstack((A_eq, new_A_eq_row))
            b_eq = np.append(b_eq, new_b_eq_row)

            # increment iteration counter
            iter_num += 1

    def var(self, alpha: float) -> float:
        """Function to return the value at risk of the empirical portfolio distribution for a given alpha"""
        return np.percentile(self.W_sort, alpha * 100, method='median_unbiased')

    def cvar(self, alpha: float) -> float:
        """Function to return the conditional value at risk of the empirical portfolio distribution for a given alpha"""
        return np.mean(self.W_sort[self.W_sort <= self.var(alpha)])

    def moment(self, n: int) -> float:
        """Function to return the nth central moment of the empirical portfolio distribution"""
        return stats.moment(self.W, moment=n)

    def get_weights(self) -> [list[str], list[float]]:
        """Function to return stocks and associated percentage weights from optimal portfolio"""
        indices = np.nonzero(self.w)
        return np.array(self.tickers)[indices], self.w[indices]

    def get_shares(self, cash):
        """Function to return stocks and associated number of shares from optimal portfolio"""
        indices = np.nonzero(self.w)
        return np.array(self.tickers)[indices], self.w[indices] * cash / self.latest_prices[indices]

    def plot_cdf(self, synthetic: bool = False) -> None:
        """Function to plot the cumulative density function for each portfolio"""
        plt.figure()
        plt.step(self.W_sort, self.Y, where='mid', label='Optimal Portfolio')
        plt.step(self.B_original_sort, self.Y, where='mid', label='Benchmark')
        if synthetic:
            plt.step(self.B_sort, self.Y, where='mid', label='Synthetic Benchmark')
        plt.xlabel('Return %')
        plt.ylabel('Cumulative Density')
        plt.title('Comparison of Optimal and Benchmark CDF\'s')
        plt.legend()
        plt.tight_layout()

    def plot_pmf(self, synthetic: bool = False) -> None:
        """Function to plot the probability mass function for each portfolio"""
        plt.figure()
        plt.hist(self.W, bins=int(np.sqrt(self.S)), density=True, histtype='step', align='mid',
                 label='Optimal Portfolio')
        plt.hist(self.B_original, bins=int(np.sqrt(self.S)), density=True, histtype='step', align='mid',
                 label='Benchmark')
        if synthetic:
            plt.hist(self.B, bins=int(np.sqrt(self.S)), density=True, histtype='step', align='mid',
                     label='Synthetic Benchmark')
        plt.xlabel('Return %')
        plt.ylabel('Probability Mass')
        plt.title('Comparison of Optimal and Benchmark PMF\'s')
        plt.legend()
        plt.tight_layout()

    def plot_returns(self, synthetic: bool = False) -> None:
        """Function to plot the cumulative returns of each portfolio over the in sample history"""
        plt.figure()
        plt.plot(self.t, np.cumprod(np.insert(self.W + 1, 0, 1)) - 1, label='Optimal Portfolio')
        plt.plot(self.t, np.cumprod(np.insert(self.B_original + 1, 0, 1)) - 1, label='Benchmark')
        if synthetic:
            plt.plot(self.t, np.cumprod(np.insert(self.B + 1, 0, 1)) - 1,
                     label='Synthetic Benchmark')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.title('Comparison of Optimal and Benchmark Cumulative Returns (In Sample)')
        plt.legend()
        plt.tight_layout()

    def plot_weights(self, N: int = 25) -> None:
        """Function to plot the top N ranked stocks and weights of the optimal portfolio"""
        N = min(self.n, N)
        plt.figure()
        x_data, y_data = zip(*sorted(list(zip(self.tickers, self.w)), key=lambda x: x[1], reverse=True))
        plt.bar(x_data[:N], y_data[:N], label='Ranked Stock Weights')
        plt.axhline(1 / self.n, linestyle='--', c='blue', label='Equal Weighting')
        plt.xlabel('Tickers')
        plt.ylabel('Weight')
        plt.title(f'SSD Portfolio Weights (Top {N})')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
    
