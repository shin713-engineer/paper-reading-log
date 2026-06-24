# Paper Title
Adam and optimizaiton
## 1. Paper Information
- Title: ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION
- Authors: Diederik P. Kingma, Jimmy Lei Ba
- Year: 2017-01-30
- Topic: Stochastic optimization
- Link: http://arxiv.org/abs/1412.6980

## 2. One-line Summary



## 3. Problem

to control high level parameters, high order optimization is too expensive and memory cost is critical. 
also we need to control noisy data and sparse gradient. 

## 4. Key Idea

adam combines two algorithm and makes efficient first order optimization. 
1. AdaGrad : controls sparse gradient
2. RMsprop : in non-stationary
it combines two alogorithms above

adam has adaptive learning rate so it applies individual learning rate for each parameter

bias correction 


## 5. Method / Architecture

## Algorithm

The Adam algorithm proceeds as follows.

1. Compute the gradient at time step $t$:

$$
g_t = \nabla_\theta f_t(\theta_{t-1})
$$

2. Update the first and second moment estimates:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

3. Apply bias correction:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

4. Update the parameter:

$$
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

## 6. Experiments and Results
## 7. Limitations
## 8. Connection to My Research
