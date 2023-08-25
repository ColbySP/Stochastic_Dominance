# imports
from scipy.optimize import linprog
from Synthetic import Synthetic
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import time

plt.style.use('ggplot')


class SSD_Portfolio:
    __slots__ = ('tickers', 'benchmark', 'start', 'end', 't', 'R', 'B_original', 'B', 'W', 'S', 'n', 'v', 'w',
                 'B_sort', 'W_sort', 'B_original_sort', 'Y')

    def __init__(self, tickers: list or str, benchmark: str, start: str, end: str):
        # check if tickers is an existing index or string of stocks instead of list
        if isinstance(tickers, str) and tickers not in ['S&P500']:
            self.tickers = tickers.split(' ')
        else:
            self.tickers = tickers

        # save benchmark ticker
        self.benchmark = benchmark

        # define start and end dates
        self.start = start
        self.end = end

        # placeholder for timestamps
        self.t = None

        # download data and save returns as numpy arrays
        print('Downloading portfolio components:')
        self.R = self.download_returns(symbols=self.tickers)
        print('\nDownloading benchmark:')
        self.B_original = self.download_returns(symbols=self.benchmark)
        self.B_original_sort = np.sort(self.B_original)

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

    def download_returns(self, symbols: list or str) -> np.array:
        """Function to download, clean, and prep stock data from online database"""
        # define common index components
        if symbols == 'S&P500':
            tickers = """
            A,AAL,AAP,AAPL,ABBV,ABC,ABT,ACGL,ACN,ADBE,ADI,ADM,ADP,ADSK,AEE,AEP,AES,AFL,AIG,AIZ,AJG,AKAM,ALB,ALGN,ALK,
            ALL,ALLE,AMAT,AMCR,AMD,AME,AMGN,AMP,AMT,AMZN,ANET,ANSS,AON,AOS,APA,APD,APH,APTV,ARE,ATO,ATVI,AVB,AVGO,AVY,
            AWK,AXON,AXP,AZO,BA,BAC,BALL,BAX,BBWI,BBY,BDX,BEN,BF.B,BG,BIIB,BIO,BK,BKNG,BKR,BLK,BMY,BR,BRK.B,BRO,BSX,BWA,
            BXP,C,CAG,CAH,CARR,CAT,CB,CBOE,CBRE,CCI,CCL,CDAY,CDNS,CDW,CE,CEG,CF,CFG,CHD,CHRW,CHTR,CI,CINF,CL,CLX,CMA,
            CMCSA,CME,CMG,CMI,CMS,CNC,CNP,COF,COO,COP,COST,CPB,CPRT,CPT,CRL,CRM,CSCO,CSGP,CSX,CTAS,CTLT,CTRA,CTSH,CTVA,
            CVS,CVX,CZR,D,DAL,DD,DE,DFS,DG,DGX,DHI,DHR,DIS,DLR,DLTR,DOV,DOW,DPZ,DRI,DTE,DUK,DVA,DVN,DXC,DXCM,EA,EBAY,
            ECL,ED,EFX,EG,EIX,EL,ELV,EMN,EMR,ENPH,EOG,EPAM,EQIX,EQR,EQT,ES,ESS,ETN,ETR,ETSY,EVRG,EW,EXC,EXPD,EXPE,EXR,F,
            FANG,FAST,FCX,FDS,FDX,FE,FFIV,FI,FICO,FIS,FITB,FLT,FMC,FOX,FOXA,FRT,FSLR,FTNT,FTV,GD,GE,GEHC,GEN,GILD,GIS,
            GL,GLW,GM,GNRC,GOOG,GOOGL,GPC,GPN,GRMN,GS,GWW,HAL,HAS,HBAN,HCA,HD,HES,HIG,HII,HLT,HOLX,HON,HPE,HPQ,HRL,HSIC,
            HST,HSY,HUM,HWM,IBM,ICE,IDXX,IEX,IFF,ILMN,INCY,INTC,INTU,INVH,IP,IPG,IQV,IR,IRM,ISRG,IT,ITW,IVZ,J,JBHT,JCI,
            JKHY,JNJ,JNPR,JPM,K,KDP,KEY,KEYS,KHC,KIM,KLAC,KMB,KMI,KMX,KO,KR,L,LDOS,LEN,LH,LHX,LIN,LKQ,LLY,LMT,LNC,LNT,
            LOW,LRCX,LUV,LVS,LW,LYB,LYV,MA,MAA,MAR,MAS,MCD,MCHP,MCK,MCO,MDLZ,MDT,MET,META,MGM,MHK,MKC,MKTX,MLM,MMC,MMM,
            MNST,MO,MOH,MOS,MPC,MPWR,MRK,MRNA,MRO,MS,MSCI,MSFT,MSI,MTB,MTCH,MTD,MU,NCLH,NDAQ,NDSN,NEE,NEM,NFLX,NI,NKE,
            NOC,NOW,NRG,NSC,NTAP,NTRS,NUE,NVDA,NVR,NWL,NWS,NWSA,NXPI,O,ODFL,OGN,OKE,OMC,ON,ORCL,ORLY,OTIS,OXY,PANW,PARA,
            PAYC,PAYX,PCAR,PCG,PEAK,PEG,PEP,PFE,PFG,PG,PGR,PH,PHM,PKG,PLD,PM,PNC,PNR,PNW,PODD,POOL,PPG,PPL,PRU,PSA,PSX,
            PTC,PWR,PXD,PYPL,QCOM,QRVO,RCL,REG,REGN,RF,RHI,RJF,RL,RMD,ROK,ROL,ROP,ROST,RSG,RTX,RVTY,SBAC,SBUX,SCHW,SEDG,
            SEE,SHW,SJM,SLB,SNA,SNPS,SO,SPG,SPGI,SRE,STE,STLD,STT,STX,STZ,SWK,SWKS,SYF,SYK,SYY,T,TAP,TDG,TDY,TECH,TEL,
            TER,TFC,TFX,TGT,TJX,TMO,TMUS,TPR,TRGP,TRMB,TROW,TRV,TSCO,TSLA,TSN,TT,TTWO,TXN,TXT,TYL,UAL,UDR,UHS,ULTA,UNH,
            UNP,UPS,URI,USB,V,VFC,VICI,VLO,VMC,VRSK,VRSN,VRTX,VTR,VTRS,VZ,WAB,WAT,WBA,WBD,WDC,WEC,WELL,WFC,WHR,WM,WMB,
            WMT,WRB,WRK,WST,WTW,WY,WYNN,XEL,XOM,XRAY,XYL,YUM,ZBH,ZBRA,ZION,ZTS
            """.replace('\n', '').replace(' ', '').replace(',', ' ')
            self.tickers = tickers.split(' ')
            symbols = self.tickers

        # download data
        df = yf.download(tickers=symbols, start=self.start, end=self.end, interval='1d', group_by='column',
                         progress=True)['Adj Close']

        if len(df.shape) > 1:
            return df.fillna(method='ffill', axis=1).pct_change(1).values[1:, :]
        else:
            self.t = df.index.values
            return df.fillna(method='ffill', axis=0).pct_change(1).values[1:]

    def adjust_benchmark(self, delta_mu: float, delta_sigma: float, delta_skew: float) -> np.array:
        """Function to generate a synthetic benchmark based on the original as outlined by Valle et al. 2017"""
        self.B = Synthetic(self.B_original, delta_mu, delta_sigma, delta_skew).gen()

    def optimize(self, time_limit: float = 60, iter_limit: int = 200) -> None:
        """Function using linear programming approach outlined by Fabian et al. 2011 to solve for optimal weights"""
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
        print('\nComputing optimal portfolio ...')
        start_time = time.time()
        iter_num = 0
        while True:

            # limit computation time
            if (time.time() - start_time) > time_limit:
                print(f'Time limit reached after {iter_num} iterations!')
                break
            if iter_num > iter_limit:
                print(f'Iteration limit reached after {time.time() - start_time:.1f} seconds!')
                break

            # run optimization
            res = linprog(c=-c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

            # ensure valid solution
            if res.x is not None:
                self.v, self.w = res.x[0], res.x[1:]

            # otherwise end iteration process
            else:
                print('Infeasible problem constraints ... returning closest solution!')
                break

            # construct the set J^*
            self.W = np.dot(self.R, self.w)
            J_star = np.argsort(self.W)
            self.W_sort = self.W[J_star]

            # compute the residuals for each tail_i
            residuals = np.zeros(shape=self.S)
            for i in range(self.S):
                residuals[i] = self.v + tau[i] - np.sum(np.dot(self.R[J_star[:i + 1], :], self.w), axis=0) / self.S

            # find the worst violation
            i_hat = np.argmax(residuals)
            if residuals[i_hat] <= 1e-7:  # set a minor tolerance
                print(f'Optimal portfolio achieved in {iter_num} iterations! ({time.time() - start_time:.1f}s)')
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

    def get_weights(self, all: bool = False) -> [list[str], list[float]]:
        """Function to return stocks and associated weights from optimal portfolio"""
        if all:
            return self.tickers, self.w
        else:
            indices = np.nonzero(self.w)
            return np.array(self.tickers)[indices], self.w[indices]

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
