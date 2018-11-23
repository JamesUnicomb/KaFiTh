# KaFiTh
Kalman Filters with Theano


## Brownian Filter
### Mackey-Glass Dataset

Applying a Brownian Filter (constant position filter) to the Mackey-Glass time series. The filter assumes that the velocity is white noise.

![](https://github.com/JamesUnicomb/KaFiTh/blob/master/Results/MackeyGlassBrownianFilter.png)


### Tracking and Predicting Brownian Motion

If we have an underlying process that is Brownian and our measurement is corrupted with white noise, then we can find a better estimate of the mean of the time series by using the Kalman Filter.

![](https://github.com/JamesUnicomb/KaFiTh/blob/master/Results/BrownianMotion1D.png)

We can also forward predict which the plot shows. As this estimate is probabilistic we can use it in other models such as VAR (value at risk models).



## Neural Network as a Motion Model

We can train an autoressive neural network to forward predict the next time series measurement as shown in the figure below.

![](https://github.com/JamesUnicomb/KaFiTh/blob/master/Results/AutoRegressiveModel.png)


### Extended Kalman Filter with AutoRegressive Motion Model

We can use the autoregressive model within an EKF framework to make better predictions about the next state of a time series.

![](https://github.com/JamesUnicomb/KaFiTh/blob/master/Results/AutoRegressiveEKF.png)


#### Forward Predtiction with Brownian Filter and AutoRegressive Extended Kalman Filters

Using a model for prediction makes the future estimates more accurate as shown in the plot below.

![](https://github.com/JamesUnicomb/KaFiTh/blob/master/Results/AutoRegressiveEKFPrediction.png)


## Matrix Square Root in Python

We can use theano to make a significant decrease in timing (although it takes time to precompile). This is useful for the implementation of the Unscented Kalman Filter or if you have to calculate the matrix square root multiple times.

### How to Use
There is currently no error handling. If the matrix is not of a particular type the result will be nan.

#### Precompile
If you want a precompiling function for calculating the matrix square root:
```
from KalmanFilters import sqrtm
A     = ~(positive semi-definite matrix - covariance matrix)
sqrtm = sqrtm() # THIS PRECOMPILES THE FUNCTION
B     = sqrtm(A)

sum(square(B * B - A)) < eps # Where eps is a small value
```

#### As a Theano variable
If you want to use this in already existing code as a Theano function:
```
import theano
from KalmanFilters import MatrixSqrt

sqrtm = MatrixSqrt()

X     = theano.tensor.fmatrix()
SqrtX = sqrtm(X)                 #This gives you the square root of the matrix X
```


### Speed Pay-Off
From the plot, we can see that the precompiled theano function is an order of magnitude (approx. 6 times faster than using a scipy function).

<img src="https://github.com/JamesUnicomb/KaFiTh/blob/master/Results/MatrixSquareRootTiming.png" width="480">
