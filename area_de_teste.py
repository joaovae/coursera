# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 15:26:36 2020

@author: JOAO VICTOR
"""

import pandas as pd
import numpy as np

data_ME = pd.read_csv("Portfolios_Formed_on_ME_monthly_EW.csv", index_col = 0)

def returns_anual(data_col):
    returns = data_col / 100

    n_months = returns.shape[0]
    returns_annualized = (returns + 1).prod()**(12/n_months)
    return returns_annualized

lo_20 = returns_anual(data_ME['Lo 20'])
hi_20 = returns_anual(data_ME['Hi 20'])

def vol_anual(data_col, n =12):
    returns = data_col / 100

    vol_annualized = returns.std()*np.sqrt(n)

    return vol_annualized


vol_lo_20 = vol_anual(data_ME['Lo 20'])
vol_hi_20 = vol_anual(data_ME['Hi 20'])

data_ME.index = pd.to_datetime(data_ME.index, format="%Y%m")
data_ME.index = data_ME.index.to_period("M")

lo_20_date = returns_anual(data_ME['1999':'2015']['Lo 20'])
hi_20_date = returns_anual(data_ME['1999':'2015']['Hi 20'])

vol_lo_20_date = vol_anual(data_ME['1999':'2015']['Lo 20'])
vol_hi_20_date = vol_anual(data_ME['1999':'2015']['Hi 20'])


def drawdown(data_col):
    rets = data_col / 100
    wealth_index = 1000*(1 + rets).cumprod()

    previous_peak = wealth_index.cummax()

    drawdown = (wealth_index - previous_peak) / previous_peak
    return drawdown

drawdown_lo20 = drawdown(data_ME['Lo 20'])

max_drawdown = drawdown_lo20['1999':'2015'].min()
time_md = drawdown_lo20['1999':'2015'].idxmin()
print(time_md)

month_max_drawdown = drawdown(data_ME['1999':'2015']['Lo 20']).idxmin()

drawdown_hi20 = drawdown(data_ME['Hi 20'])

max_drawdown_hi = drawdown_hi20['1999':'2015'].min()
time_md_hi = drawdown_hi20['1999':'2015'].idxmin()
print(time_md_hi)

month_max_drawdown = drawdown(data_ME['1999':'2015']['Lo 20']).idxmin()


edhec_data = pd.read_csv("edhec-hedgefundindices.csv", index_col = 0)
hfi = edhec_data
hfi.index = pd.to_datetime(hfi.index, format="%d/%m/%Y")

hfi.index = hfi.index.to_period('M')

def semideviation(data):
    is_negative = data < 0
    return data[is_negative].std(ddof = 0)

semi_indexs = semideviation(hfi['2009':'2018'])
max_semid_index = semi_indexs.idxmax()
min_semid_index = semi_indexs.idxmin()

hfi_analysis = hfi['2000':]

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4


from scipy.stats import norm

def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )

    return -(r.mean() + z*r.std(ddof=0))

def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

var_modified = var_gaussian(hfi_analysis["Distressed Securities"], level = 1, modified = True)

historical_var = var_historic(hfi_analysis["Distressed Securities"], level = 1)

ind =  pd.read_csv("data/ind30_m_vw_rets.csv", header=0, index_col=0)/100
ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
ind.columns = ind.columns.str.strip()

def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    return (weights.T @ covmat @ weights)**0.5


def plot_ef2(n_points, er, cov):
    """
    Plots the 2-asset efficient frontier
    """
    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style=".-")


from scipy.optimize import minimize

def minimize_vol(target_return, er, cov):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x


def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol

    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x


def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


def optimal_weights(n_points, er, cov):
    """
    Returns a list of weights that represent a grid of n_points on the efficient frontier
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


def plot_ef(n_points, er, cov, style='.-', legend=False, show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False):
    """
    Plots the multi-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)
    if show_cml:
        ax.set_xlim(left = 0)
        # get MSR
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # add EW
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # add EW
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)

        return ax



er4 = ind_analysis[asset_analysis]
n = er4.shape[0]
w_ew = np.repeat(1/n, n)
r_ew = portfolio_return(w_ew, er4)
vol_ew = portfolio_vol(w_ew, er4.cov)



def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

ind_analysis = ind["2013":"2017"]
ind_analysis_anual = annualize_rets(ind["2013":"2017"], 12)
cov = ind_analysis.cov()
asset_analysis = ["Books", "Steel", "Oil", "Mines"]

question5 = msr(0.1, ind_analysis_anual[asset_analysis], cov.loc[asset_analysis, asset_analysis])
question5[2]

question8 = gmv(cov.loc[asset_analysis, asset_analysis])

def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    return r.std()*(periods_per_year**0.5)

question11 = portfolio_vol(question5 ,ind["2018"][asset_analysis].cov())
question11 = question11*(12**0.5)

question12 = portfolio_vol(question8 ,ind["2018"][asset_analysis].cov())
question12 = question12*(12**0.5)