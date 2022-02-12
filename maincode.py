#!/usr/bin/env python
# coding: utf-8

# In[118]:


#libraries
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib. pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import pickle


# In[119]:


data= pd.read_csv('gcdata.csv')
data.head()


# ## Calculating One Day Returns
# <font color='green'>Definition: ODR is the price of stocks at today's closure compared to the price of the same stock at yesterday's closure.</font>

# In[120]:


data['ODR'] = data.x4.pct_change()


# ### Momentum Calculation

# In[213]:


def mv(dataframe):
    n = len(dataframe)
    arr = []
    for i in range(0,5):
        arr.append('N')
    for j in range(5,n):
        mv = dataframe.x4[j] - dataframe.x4[j-5] #Equation for momentum
        arr.append(mv)
    return arr

mv = mv(data)

# add momentum to data
data['Momentum'] = mv


# ## Return of Investment
# 
# <font color='green'>Definition: Return on investment (ROI) is a performance measure used to evaluate the efficiency or profitability of an investment or compare the efficiency of a number of different investments. ROI tries to directly measure the amount of return on a particular investment, relative to the investment's cost.</font>
# 
# <font color='blue'>Formula: ROI is calculated by subtracting the initial value of the investment from the final value of the investment (which equals the net return), then dividing this new number (the net return) by the cost of the investment, then finally, multiplying it by 100.</font>
# 
# 

# In[122]:


def ROI(dataframe,n):
    l = len(dataframe)
    arr = []
    for i in range(0,n):
        arr.append('N')
    for j in range(n,l):
        roi= (dataframe.x4[j] - dataframe.x4[j-n])/dataframe.x4[j-n] #Equation for ROI
        arr.append(roi)
    return arr

#calculating ROI for 10, 20 and 30 day periods

ROI10=ROI(data,10)
ROI20=ROI(data,20)
ROI30=ROI(data,30)


#adding all the above data to our core dataframe
data['10 Days ROI']=ROI10
data['20 Days ROI']=ROI20
data['30 Days ROI']=ROI30

data


# ## Relative strength index
# <font color='green'>Definition: The relative strength index (RSI) is a momentum indicator used in technical analysis that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset.</font>
# 
# <font color='blue'>Formula for Calculating RSI: RSI = 100 – [100 / ( 1 + (Average of Upward Price Change / Average of Downward Price Change ) ) ].</font>

# In[123]:


def RSI(dataframe,period):
    # calculating the average of upwards of last 14 days: Ct - Ct-1
    #calculating the average of downwards of last 14 days: Ct-1 - Ct
    n = len(dataframe)
    arr = []
    for i in range(0,period):
        arr.append('N')
    for j in range(period,n):
        total_upwards = 0
        total_downwards = 0
        # this will find average of upwards
        for k in range(j,j-period,-1):
            if(dataframe.x4[k-1] > dataframe.x4[k]):
                total_downwards = total_downwards + (dataframe.x4[k-1] - dataframe.x4[k])    
        avg_down = total_downwards / period
        for l in range(j,j-period,-1):
            if(dataframe.x4[l] > dataframe.x4[l-1]):
                total_upwards = total_upwards + (dataframe.x4[l] - dataframe.x4[l-1])
        avg_up = total_upwards / period
        RS = avg_up / avg_down
        RSI  = 100 - (100/(1+RS))
        arr.append(RSI)
    return arr


#calculating relative strength index for 10, 14, and 30 days periods

RSI_14 = RSI(data,14)
RSI_10 = RSI(data,10)
RSI_30 = RSI(data,30)

#adding the above RSI data to the core dataframe

data['10_day_RSI'] = RSI_10
data['14_day_RSI'] = RSI_14
data['30_day_RSI'] = RSI_30
data


# ## Exponential Moving Average
# 
# <font color='green'>Definition: The exponential moving average (EMA) is a technical chart indicator that tracks the price of an investment (like a stock or commodity) over time. The EMA is a type of weighted moving average (WMA) that gives more weighting or importance to recent price data.</font>
# 
# <font color='blue'>Formula for Calculating EMA: EMA = EMA = Price(t)×k+EMA(y)×(1−k). 
# where:
# t = today;
# y = yesterday;
# N = number of days in EMA;
# k = 2÷(N+1).
# </font>
# 
# 

# In[124]:


def EMA(dataframe, n):
    m = len(dataframe)
    arr = []
    arr.append('N')
    prevEMA = dataframe.x4[0]
    for i in range(1,m):
        close = dataframe.x4[i]
        EMA = ((2/(n+1))*close) + ((1-(2/(n+1)))*prevEMA)
        arr.append(EMA)
        prevEMA = EMA
    return arr

#Calculating EMA keeping n=12 and n=26

EMA_12 = EMA(data, 12)
EMA_26 = EMA(data, 26)

#adding EMA data to the core dataframe 

data['EMA_12'] = EMA_12
data['EMA_26'] = EMA_26
data


# ## Moving Average Convergence/Divergence
# 
# <font color='green'>Definition: It is a trading indicator used in technical analysis of stock prices, created by Gerald Appel in the late 1970s. It is designed to reveal changes in the strength, direction, momentum, and duration of a trend in a stock's price.</font>
# 
# <font color='blue'>Formula for Calculating MACD: Moving Average of EMA(n) - EMA(m2) for each row</font>

# In[125]:


# where n = 12 and m2 = 26
def MACD(dataframe):
    n = 12
    m2 = 26
    arr = []
    arr.append('N')
    ema_12 = EMA(dataframe,n)
    ema_26 = EMA(dataframe,m2)
    m = len(dataframe)
    for i in range(1,m):
        arr.append(ema_12[i] - ema_26[i])
    return arr

MACD = MACD(data)

#Add MACD to our dataframe 
data['MACD_12_26'] = MACD
data


# ## Stochastic RSI
# 
# <font color='green'>Definition: The stochastic RSI (StochRSI) is a technical indicator used to measure the strength and weakness of the relative strength indicator (RSI) over a set period of time.</font>
# 
# <font color='blue'>Formula for Calculating EMA: SRSI = (RSI_today - min(RSI_past_n)) / (max(RSI_past_n) - min(RSI_past_n).</font>
# 
# 
# 

# In[126]:


def SRSI(dataframe,n):
    m = len(dataframe)
    arr = []
    list_RSI = RSI(dataframe,n)
    for i in range(0,n):
        arr.append('N')
    for j in range(n,n+n):
        last_n = list_RSI[n:j]
        if(not(last_n == []) and not(max(last_n) == min(last_n))):
            SRSI = (list_RSI[j] - min(last_n)) / (max(last_n)- min(last_n))
            if SRSI > 1:
                arr.append(1)
            else:
                arr.append(SRSI)
        else:
            arr.append(0)
    for j in range(n+n,m):
        last_n = list_RSI[2*n:j]
        if(not(last_n == []) and not(max(last_n) == min(last_n))):
            SRSI = (list_RSI[j] - min(last_n)) / (max(last_n)- min(last_n))
            if SRSI > 1:
                arr.append(1)
            else:
                arr.append(SRSI)
        else:
            arr.append(0)
    return arr

#SRSI for 10, 14, and 30 day periods
SRSI_10 = SRSI(data,10)
SRSI_14 = SRSI(data,14)
SRSI_30 = SRSI(data,30)

#Adding SRSI to the core dataframe
data['SRSI_10'] = SRSI_10
data['SRSI_14'] = SRSI_14
data['SRSI_30'] = SRSI_30


# ## True Range
# 
# <font color='green'>Definition: True range is a technical analysis volatility indicator originally developed by J. Welles Wilder, Jr. for commodities. The indicator does not provide an indication of price trend, simply the degree of price volatility. The average true range is an N-period smoothed moving average of the true range values.</font>
# 
# <font color='blue'>Formula for Calculating True Range: TR = MAX(high[today] - close[yesterday]) - MIN(low[today] - close[yesterday])</font>

# In[127]:


def TR(dataframe,n):
    high = dataframe.x2[n]
    low = dataframe.x3[n]
    close = dataframe.x4[n-1]
    l_max = list()
    l_max.append(high)
    l_max.append(close)
    l_min = list()
    l_min.append(low)
    l_min.append(close)
    return (max(l_max) - min(l_min))

# Average True Range
# Same as EMA except use TR in lieu of close (prevEMA = TR(dataframe,14days))
def ATR(dataframe,n):
    m = len(dataframe)
    arr = []
    prevEMA = TR(dataframe,n+1)
    for i in range(0,n):
        arr.append('N')
    for j in range(n,m):
        TR_ = TR(dataframe,j)
        EMA = ((2/(n+1))*TR_) + ((1-(2/(n+1)))*prevEMA)
        arr.append(EMA)
        prevEMA = EMA
    return arr

ATR = ATR(data,14)  

#Adding ATR to the core dataframe
data['ATR_14'] = ATR


# ## Williams %R oscillator
# 
# <font color='green'>Definition: It compares a stock's closing price to the high-low range over a specific period, typically 14 days or periods. Williams %R oscillates from 0 to-100; readings from 0 to -20 are considered overbought, while readings from -80 to -100 are considered oversold.</font>
# 
# <font color='blue'>Formula for Calculating Williams %R oscillator: %R = (Highest High - Close)/(Highest High - Lowest Low) * -100 </font>

# In[128]:


def Williams(dataframe,n):
    m = len(dataframe)
    arr = []
    for i in range(0,n-1):
        arr.append('N')
    for j in range(n-1,m):
        maximum = max(data.x2[(j-n+1):j+1])
        minimum = min(data.x3[(j-n+1):j+1])
        val = (-100)*(maximum-dataframe.x4[j])/(maximum-minimum)
        arr.append(val)
    return arr


williams = Williams(data,14)

#Add Williams%R to the core dataframe
data['Williams'] = williams


# ## Commodity Channel Index (CCI)
# 
# <font color='green'>Definition: The Commodity Channel Index (CCI) is calculated by determining the difference between the mean price of a security and the average of the means over the period chosen. This difference is compared to the average difference over the time period.
# </font>
# 
# <font color='blue'>Formula for Calculating CCI: CCI = (Typical Price - 20-period SMA of TP) / (.015 x Mean Deviation) | Typical Price (TP) = (High + Low + Close)/3 | Constant = 0.015</font>

# In[129]:


def CCI(dataframe,n):
    m = len(dataframe)
    arr = []
    tparr = []
    for i in range(0,n-1):
        arr.append('N')
        tp = (dataframe.x2[i]+dataframe.x3[i]+dataframe.x4[i])/3
        tparr.append(tp)
    for j in range(n-1,m):
        tp = (dataframe.x2[j]+dataframe.x3[j]+dataframe.x4[j])/3
        tparr.append(tp) 
        tps = np.array(tparr[(j-n+1):(j+1)])
        val = (tp-tps.mean())/(0.015*tps.std())
        arr.append(val)
    return arr

cci = CCI(data,20) 

#Adding CCI to the core dataframe
data['CCI'] = cci


# In[130]:


#double check that the dataframe has all 22 features
data.shape


# ## Normalizing the data

# In[131]:


def normalize(dataframe):
    for column in dataframe:
        dataframe[column]=((dataframe[column]-dataframe[column].mean())/dataframe[column].std())


# # Taking only positive values for running Multinomial Naive Bayes

# In[132]:


def positivevalues(dataframe):
    for column in dataframe:
        if (dataframe[column].min())<0:
            dataframe[column]=(dataframe[column]-dataframe[column].min())
data


# ## Cleaning the Data

# In[133]:


#Remove the first 30 index which could have a value 'N'
final_data = data.drop(data.index[0:30])

#Remove the last row of data because class has value 'N'
final_data = final_data.drop(final_data.index[-1])

#Remove the feature columns to improve the algorithm
final_data = final_data.drop(['y'], axis=1)

#Remove 'High' and 'Low' columns to improve the algorithm
final_data = final_data.drop(['x2','x3'], axis=1)

#check the features that remain in our algorithm 
final_data.head()


# In[186]:


#Normalize the data that we have filtered
normalize(final_data)


# ## Selecting feature data and label data 

# In[199]:


X = final_data
y = data.y[30:-1]


# ## Splitting into test and train set

# In[200]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


# ## Running the Logistic Regression Model

# In[201]:


rishismodellr = LogisticRegression()
rishismodellr.fit(X,y)


# ## Saving the Logistic Regression Model

# In[203]:


filename = 'finalrishismodelLR.sav'
pickle.dump(rishismodellr, open(filename, 'wb'))


# In[160]:


loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)


# ## Final Test Data

# In[190]:


y_predictions_LR = rishismodellr.predict(X)


# In[192]:


print (metrics.accuracy_score(y, y_predictions_LR)) 


# ## Accuracy score of our predicted y for the Logistic Regression Model
# 

# In[142]:


print(classification_report(y_test,y_predictions_LR))


# ## Running the Gaussian Naive Bayes Model

# In[143]:


rishismodelGNB = GaussianNB()
rishismodelGNB.fit(X_train,y_train)


# In[144]:


y_predictions_GNB = rishismodelGNB.predict(X_test)


# ## Accuracy score of our predicted y fot the Gaussian Niave Byes Model

# In[145]:


print (metrics.accuracy_score(y_test, y_predictions_GNB)) 


# In[146]:


print(classification_report(y_test,y_predictions_GNB))


# ## Accuracy score of our predicted y fot the KNN Model
# 

# # So the Logstic Regression Model is giving the highest Accuracy Score of 54-55%

# In[ ]:




