# imports
from SSD_Opt import SSD_Portfolio
import matplotlib.pyplot as plt

# instantiate model by defining directories for asset returns and benchmark return data
model = SSD_Portfolio(return_dir="sample_ret.csv", benchmark_dir="sample_ref.csv")

# create a synthetic benchmark to optimize against
# below introduces a +5% relative return, -20% relative variance, +0% relative skew synthetic over original benchmark
model.adjust_benchmark(delta_mu=0.05, delta_sigma=-0.2, delta_skew=0)

# compute optimal portfolio within a time (sec) or iteration limit (#)
model.optimize(time_limit=60, iter_limit=500)

# plot the pmf of returns for each portfolio
model.plot_pmf(synthetic=True)

# plot the cdf of returns for each portfolio
model.plot_cdf(synthetic=True)

# plot the cumulative returns for each portfolio (in-sample)
model.plot_returns(synthetic=True)

# plot the top N weighted stocks in optimal portfolio
model.plot_weights(N=25)

# retrieve used stocks and associated weights from optimal portfolio
tickers, weights = model.get_weights()
print(list(zip(tickers, weights)))

# display figures
plt.show()
