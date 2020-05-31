# 1. Anomaly search

## 1.1 Statistical Process Control:

* understanding of process and its limits
* data cleansing from specific causes of variability (holidays, weekends, other indicators specific to your business)
* monitoring of the current system on charts for outliers (time series, mean, mean +/- 2 standard deviations)

## 1.2 Kolmogorov–Smirnov test

**Idea**: compare two windows of time series betwenn each other, if thay have different distributions, we suppose, that one of them has anomaly.

**+**: 

* works for periodical data, if the period is shorter than the window length
* robust to outliers

**-**:

* false positives in the presence of trends and seasonality
* need a lot of date within one window (more then 100)

# 2. Windows Estimations

**Idea**: calculate metric for a certain window of time series

**What for?**

* smooth observations
* highlight trends

**+**:

* easy to use
* a lot of false positive results

### 2.1 Moving Average:

$$\hat{y}_{t} = \frac{1}{k} \displaystyle\sum^{k}_{n=1} y_{t-n}$$

**-**:

* prediction for one step above

### 2.2 Weighted Moving Average:

$\hat{y}_{t} = \displaystyle\sum^{k}_{n=1} \omega_n y_{t+1-n}$

**+**:

* the ability to take into account old and new data with different weights

**-**:

* prediction for one step above

### 2.3 Exponential Smoothing

$$\hat{y}_{t} = \alpha \cdot y_t + (1-\alpha) \cdot \hat y_{t-1} $$

**-**:

* bad for timeseries with uptrend/downtrend, because estimation will always fall behind real data

### 2.4 Double Exponential Smoothing

$$\ell_x = \alpha y_x + (1-\alpha)(\ell_{x-1} + b_{x-1})$$

$$b_x = \beta(\ell_x - \ell_{x-1}) + (1-\beta)b_{x-1}$$

$$\hat{y}_{x+1} = \ell_x + b_x$$

**-**:

* does not take seasonality into account

### 2.5 Triple Exponential Smoothing a.k.a. Holt-Winters

$$\ell_x = \alpha(y_x - s_{x-L}) + (1-\alpha)(\ell_{x-1} + b_{x-1})$$

$$b_x = \beta(\ell_x - \ell_{x-1}) + (1-\beta)b_{x-1}$$

$$s_x = \gamma(y_x - \ell_x) + (1-\gamma)s_{x-L}$$

$$\hat{y}_{x+m} = \ell_x + mb_x + s_{x-L+1+(m-1)modL}$$

**+**:

* takes into account trend and seasonality
* can predict m steps ahead

**Brutlag Method (produce confidence intervals):**

$$\hat y_{max_x}=\ell_{x−1}+b_{x−1}+s_{x−T}+m⋅d_{t−T}$$

$$\hat y_{min_x}=\ell_{x−1}+b_{x−1}+s_{x−T}-m⋅d_{t−T}$$

$$d_t=\gamma∣y_t−\hat y_t∣+(1−\gamma)d_{t−T},$$

## 3. Cross Validation for Timeseries:

Idea of this approach is shown here (Nested cross validation) : https://towardsdatascience.com/time-series-nested-cross-validation-76adba623eb9

## 4. Econometric approach

To apply econometrics approaches, our data should satisfy some theoretical requirements:

* Stationarity -  if a process is *stationary*, that means it does not change its statistical properties over time, namely its mean and variance. (The constancy of variance is called homoscedasticity). The covariance function does not depend on time; it should only depend on the distance between observations. 

If data is stationary we can apply following models:

* SARIMA
* ARIMA

## 5. How to check the stationarity of a series?

* Dickey-Fuller test
* plot autocorrelation
* plot partial autocorrelation

## 6. How to get rid of non-stationary?

* apply box-coke transformation to data: $exp(\frac{1}{\lambda}(\lambda y+1))$.
* apply seasonal differences (from current value substract season value)
* apply shift to 1

## 7. Prophet by Facebook

**+**:

* works well out of the box

**-**:

* not very accurate
* works strange with non-standard periods (seconds, minutes, hours): thus, if you want to use it, aggregate such periods

## 8. Classical ML algorithms with good features from timeseries

* Time series lags
* Window statistics:
    - Max/min value of series in a window
    - Average/median value in a window
    - Window variance
    - etc.
* Date and time features:
    - Minute of an hour, hour of a day, day of the week, and so on
    - Is this day a holiday? Maybe there is a special event? Represent that as a boolean feature
* Target encoding 
* Forecasts from other models (note that we can lose the speed of prediction this way)
* Trend encoding

Use features in classical ML algorithm (xgboost, linaer regression). Note, that xgboost for timeseries tends to overfitting and does not take into account trends. That's why good idea will be to use trand encoding.



















