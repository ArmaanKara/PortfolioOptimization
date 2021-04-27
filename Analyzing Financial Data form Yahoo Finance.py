#!/usr/bin/env python
# coding: utf-8

# <h2>Project Outline</h2>
# 
# **Imported/Downloaded the Data**
# - Accessed financial data using one of the APIs accessed through the pandas-datareader package or from websites such as Yahoo Finance.
# - Loaded the data into a pandas DataFrame so that it can easily view and manipulate the data.<br />
# 
# **Calculated Financial Statistics**
# - Calculated some of the financial statistics (i.e. variance, standard deviation, linear regression) that I have learned about to gain insights into the stocks and how they relate to each other. What are the returns of the stocks over different time periods? How risky are each of the stocks when compared to each other? Do the returns of the stocks correlate with each other, or are they diversified?<br />
# 
# **Optimized Portfolio**
# - Performed a mean-variance portfolio optimization that shows the efficient frontier for the group of stocks you have selected. If the investor is less risky, how should they allocate their funds across the portfolio? If the investor is more risky, how should they allocate their funds? Indicate multiple investment options at different risk levels and specify the returns.<br />
# 

# In[27]:


import pandas as pd
import numpy as np
import seaborn as sn
import pandas_datareader.data as web
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime
import random


# In[28]:


import cvxopt as opt
from cvxopt import blas, solvers


# <h2>Portfolio</h2>

# Tesla Motors (TSLA), Palantir Technolgies Inc.(PLTR), Johnson & Johnson (JNJ), Disney (DIS)

# <h2>Stock Analysis</h2>

# - Pulling stock data from Yahoo API for stocks above.
# - Start date is from when PLTR went public.

# In[3]:


symbols = ['TSLA', 'PLTR', 'JNJ', 'DIS']
start_date = datetime(2020, 9, 30)
end_date = datetime(2021, 5, 16)
stock_data = web.get_data_yahoo(symbols, start_date, end_date)
stock_data


# In[4]:


stock_data_df = pd.DataFrame(stock_data['Adj Close'].to_records())
stock_data_df.columns = [hdr.replace("('symbols', ", "date.").replace(")", "")                      for hdr in stock_data_df.columns]
stock_data_df.rename(columns={'Date' : 'Date(Y/M/D)'}, inplace=True)
stock_data_df


# <h2>Plot of the Adjusted Close Price over Time</h2>

# In[5]:


plt.plot(stock_data.index, stock_data['Adj Close'])
plt.legend(symbols)
plt.xlabel('Date (Y/M/D)')
plt.ylabel('Adjusted Price at Close')
plt.title('Adjusted Closing Price over Time')
plt.show()


# In[6]:


selected=list(stock_data.columns[0:])

returns_monthly = stock_data[selected].pct_change()


# <h2>Plot of the daily simple rate of return over time</h2>

# In[7]:


plt.plot(returns_monthly.index, returns_monthly['Adj Close'])
plt.legend(symbols)
plt.xlabel('Date')
plt.ylabel('Monthly Returns')
plt.title('Stocks Monthly Returns since PLTR inception')
plt.savefig('Capstone_1_QR.png')
plt.show()


# In[8]:


plt.figure(figsize=(12,10))
ax1 = plt.subplot(2,2,1)
plt.plot(returns_monthly.index, returns_monthly["Adj Close"]["TSLA"], color='blue')
plt.xlabel("Date")
plt.ylabel("Monthly Returns")
plt.title("Tesla Motors (TSLA) Monthly Returns Over Time")

ax1 = plt.subplot(2,2,2)
plt.plot(returns_monthly.index, returns_monthly["Adj Close"]["PLTR"], color='orange')
plt.xlabel("Date")
plt.ylabel("Monthly Returns")
plt.title("Palantir Technologies(PLTR) Monthly Returns Over Time")

ax3 = plt.subplot(2,2,3)
plt.plot(returns_monthly.index, returns_monthly["Adj Close"]["JNJ"], color='green')
plt.xlabel("Date")
plt.ylabel("Monthly Returns")
plt.title("Johnson&Johnson (JNJ) Monthly Returns Over Time")

ax4 = plt.subplot(2,2,4)
plt.plot(returns_monthly.index, returns_monthly["Adj Close"]["DIS"], color='red')
plt.xlabel("Date")
plt.ylabel("Monthly Returns")
plt.title("Disney (DIS) Monthly Returns Over Time")
plt.show()


# <h2>Plot of the mean of each stock's daily simple rate of return</h2>

# In[9]:


expected_returns = returns_monthly.mean()


# In[10]:


fig, ax = plt.subplots(figsize= (10,5))
plt.bar(range(len(expected_returns['Adj Close'])), expected_returns['Adj Close'])
ax.set_xticks(range(len(expected_returns['Adj Close'])))
ax.set_xticklabels(['TSLA', 'PLTR', 'JNJ', 'DIS'])
plt.xlabel('Company')
plt.ylabel('Expected Returns')
plt.title('Adj. Close Expected Returns in Time Window')
plt.savefig('Capstone_1_Expected_returns.png')
plt.show()


# In[11]:


print('TSLA: ', expected_returns['Adj Close']['TSLA'])
print('PLTR: ', expected_returns['Adj Close']['PLTR'])
print('JNJ: ', expected_returns['Adj Close']['JNJ'])
print('DIS: ', expected_returns['Adj Close']['DIS'])


# In[12]:


expected_returns_var = returns_monthly.var()


# In[13]:


expected_returns_var


# <h2>Plot of the variance of each stock's daily simple rate of return</h2>

# In[14]:


fig, ax = plt.subplots(figsize= (10,5))
plt.bar(range(len(expected_returns_var['Adj Close'])), expected_returns_var['Adj Close'])
ax.set_xticks(range(len(expected_returns_var['Adj Close'])))
ax.set_xticklabels(['TSLA', 'PLTR', 'JNJ', 'DIS'])
plt.xlabel('Company')
plt.ylabel('Variance')
plt.title('Adj. Close Variance for Stocks')
plt.savefig('Capstone_1_variance.png')
plt.show()


# In[15]:


print('TSLA: ', expected_returns_var['Adj Close']['TSLA'])
print('PLTR: ', expected_returns_var['Adj Close']['PLTR'])
print('JNJ: ', expected_returns_var['Adj Close']['JNJ'])
print('DIS: ', expected_returns_var['Adj Close']['DIS'])


# Palantir Technologies has the highest variance from Septermber 30, 2020 to April 16, 2021 indicating it is most likely the riskier investment. Johnson & johnson shows the least variance indicating it is most likely the least risky investment.

# <h2>Plot of the standard deviation of each stock's daily simple rate of return</h2>

# In[16]:


std_returns = returns_monthly.std()


# In[17]:


std_returns


# In[18]:


fig, ax = plt.subplots(figsize= (10,5))
plt.bar(range(len(std_returns['Adj Close'])), std_returns['Adj Close'])
ax.set_xticks(range(len(std_returns['Adj Close'])))
ax.set_xticklabels(['TSLA', 'PLTR', 'JNJ', 'DIS'])
plt.xlabel('Company')
plt.ylabel('Standard Deviation')
plt.title('Adj. Close Returns Standard Deviation for Stocks')

plt.show()


# In[19]:


print('TSLA: ', std_returns['Adj Close']['TSLA'])
print('PLTR: ', std_returns['Adj Close']['PLTR'])
print('JNJ: ', std_returns['Adj Close']['JNJ'])
print('DIS: ', std_returns['Adj Close']['DIS'])


# Palantir is the most volatile stock but it also has the most mean return. This means that the the stock on average is making more money than the others. Tesla is a close second to Palantir. Johnson & Johnson is the least volatile of the bunch and has the least mean return.

# In[29]:


corr_returns = returns_monthly['Adj Close'].cov()
corr_returns


# Conclusion so far PLTR and TSLA appear to be postively correlated to each other. TSLA and DIS appear to be negatively correlated and PLTR and JNJ appear to also be negatively correlated with eachother. Overall, the stocks are only minimally correlated to eachother which helps diversification.

# In[35]:


corr_matrix = pd.DataFrame(corr_returns, columns=['TSLA','PLTR','JNJ', 'DIS'])

corrMatrix = corr_matrix.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()


# <h2>Portfolio Optimization</h2>

# In[40]:


expected_returns = returns_monthly['Adj Close'].mean()
corr_returns = returns_monthly['Adj Close'].cov()


# In[43]:


def return_portfolios(expected_returns, cov_matrix):
    np.random.seed(1)
    port_returns = []
    port_volatility = []
    stock_weights = []
    
    selected = (expected_returns.axes)[0]
    
    num_assets = len(selected) 
    num_portfolios = 5000
    
    for single_portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, expected_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)
    
    portfolio = {'Returns': port_returns,
                 'Volatility': port_volatility}
    for counter,symbol in enumerate(selected):
        portfolio[f"{symbol} Weight"] = [Weight[counter] for Weight in stock_weights]
    
    df = pd.DataFrame(portfolio)
    
    column_order = ['Returns', 'Volatility'] + [f"{stock} Weight" for stock in selected]
    
    df = df[column_order]
   
    return df


# In[44]:


random_portfolios = return_portfolios(expected_returns, corr_returns) 
print(random_portfolios.head().round(4))


# In[46]:


random_portfolios.plot.scatter(x='Volatility', y='Returns', figsize=(10,5))
plt.title('Efficient Frontier')
plt.ylabel('Expected Returns')
plt.xlabel('Volatility (Standard Deviation)')
plt.show()


# In[47]:


returns_monthly = returns_monthly['Adj Close']


# In[48]:


n = returns_monthly.shape[1]
n


# In[49]:


def optimal_portfolio(returns):
    n = returns.shape[1]
    returns = np.transpose(returns.values)

    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks


# In[50]:


weights, returns, risks = optimal_portfolio(returns_monthly[1:])


# In[54]:


print(weights)
symbols


# In[56]:


companies = ['Tesla Motors', 'Palantir Technologies', 'Johnson & Johnson', 'Disney']
companies


# In[57]:


a = {'Company':companies, 'Symbol':symbols, 'Weight':[item for sublist in weights for item in sublist], 'Perc_Weight':["{:.1%}".format(item) for sublist in weights for item in sublist]}
print(a)


# In[60]:


df = pd.DataFrame(a)


# In[61]:


df


# In[62]:


ax = plt.subplot()
plt.bar(range(len(df['Weight'])), df['Weight'])
ax.set_xticks(range(len(df['Weight'])))
ax.set_xticklabels(df['Symbol'])
plt.xlabel('Company Stock Symbol')
plt.ylabel('Weight')
plt.title('Optimum Weight of Investment per Company Stock in Time Window')
plt.show()


# The plot above appears to recommend your portfolio has a very large weight (99.99%, nearly 100%) of Palantir and about the same distribution for the other five stocks. Otimized for historical risk and return data. Even though Palantir has the highest volatility, it has the highest return in the optimized portfolio.

# In[63]:


print(risks)


# In[64]:


single_asset_std=np.sqrt(np.diagonal(corr_returns))


# In[65]:


single_asset_std


# In[66]:


random_portfolios.plot.scatter(x='Volatility', y='Returns', fontsize=12, figsize=(10,5))
try:
    plt.plot(risks, returns, 'y-o')
except:
    pass
plt.scatter(single_asset_std,expected_returns,marker='X',color='red',s=200)
for i, txt in enumerate(symbols):
    plt.annotate(txt, (single_asset_std[i],expected_returns[i]), size=12)
for xc in single_asset_std:
    plt.axvline(x=xc, color='red')
plt.ylabel('Expected Returns',fontsize=14)
plt.xlabel('Volatility (Std. Deviation)',fontsize=14)
plt.title('Asset Port. 1 Efficient Frontier', fontsize=24)
plt.savefig("CapstoneOpt1_EfficientFrontier_withline.png")
plt.show()


# Each blue dot shows the wide range of portfolios according to expected returns and volatility. The efficient frontier yellow line is on top of the top-left edge of the portfolio range. This line falls on the portfolios that maximize the expected return at all risks, and minimize the risk at all expected returns.
# The vertical red lines in the figure to the right display the standard deviation of each asset. Notice, there are many portfolios (blue dots) with volatility lower than the least volatile asset. This feature results from having multiple, uncorrelated assets in the same portfolio.

# In[73]:


expected_returns


# Palantir and Tesla are risky (high standard dev) stocks with a very high expected return.
# From lessons but applies to my data too: The vertical red lines in the figure to the right display the standard deviation of each asset. Notice, there are a few portfolios (blue dots) with volatility at/lower than the least volatile asset. This feature results from having multiple, uncorrelated assets in the same portfolio.

# In[75]:


random_portfolios.sort_values(["PLTR Weight"], ascending=False)


# In[77]:


random_portfolios.sort_values(["TSLA Weight"], ascending=False)


# In[78]:


random_portfolios.sort_values(["DIS Weight"], ascending=False)


# In[79]:


random_portfolios.sort_values(["JNJ Weight"], ascending=False)


# Sorting through the random portfolios, it appears that many portoflios with large weights of Palantir and Tesla have the highest returns and highest volatility.

# In[ ]:




