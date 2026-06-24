# Paper Title
Adam and optimizaiton
## 1. Paper Information
- Title: ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION
- Authors: Diederik P. Kingma, Jimmy Lei Ba
- Year: 2017-01-30
- Topic: Stochastic optimization
- Link: http://arxiv.org/abs/1412.6980

## 2. One-line Summary

Adam is optimizer that updates each parameter using $m_t$ and $v_t$

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

## goal 

minimizing loss funciton 

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

moving average of gradient 

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

moving average of gradient size 

3. Apply bias correction:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

bias-corrected estimates 

4. Update the parameter:

$$
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

direction: like momentum

$\alpha$ : entire step size

$\epsilon$ : to prevent dividing by zero 

## Initialization Bias Correction

Adam initializes the first and second moment vectors as zero:

$$
m_0 = 0, \quad v_0 = 0
$$

Because of this zero initialization, the moving averages $m_t$ and $v_t$ are biased toward zero during the early timesteps.

For the second moment estimate, Adam uses:

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

By expanding the recurrence, we obtain:

$$
v_t = (1-\beta_2)\sum_{i=1}^{t}\beta_2^{t-i}g_i^2
$$

If the second moment of the gradient is approximately stationary, then:

$$
E[v_t] = E[g_t^2](1-\beta_2^t)
$$

Thus, $v_t$ underestimates the true second moment by the factor $(1-\beta_2^t)$. Adam corrects this initialization bias by computing:

$$
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

Similarly, the first moment estimate is corrected as:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}
$$

This correction is especially important in the early stages of training, because without it the second moment estimate can be too small, leading to unstable or overly large parameter updates.

## Convergence Analysis

The paper analyzes Adam using the online convex optimization framework.

At each timestep $t$, the optimizer chooses a parameter vector $\theta_t$ and then evaluates it on a convex cost function $f_t$.

The regret is defined as:

$$
R(T) = \sum_{t=1}^{T} \left[ f_t(\theta_t) - f_t(\theta^*) \right]
$$

where

$$
\theta^* = \arg\min_{\theta \in X} \sum_{t=1}^{T} f_t(\theta)
$$

This measures how much worse Adam performs compared to the best fixed parameter chosen in hindsight.

The main theoretical result is that Adam achieves a regret bound of:

$$
R(T) = O(\sqrt{T})
$$

Therefore, the average regret satisfies:

$$
\frac{R(T)}{T} = O\left(\frac{1}{\sqrt{T}}\right)
$$

and converges to zero as $T \to \infty$.

This means that, under the assumptions of convex cost functions, bounded gradients, and bounded parameter distances, Adam is theoretically guaranteed to converge in the online convex optimization sense.

However, this result does not directly prove convergence for general deep neural networks, because deep learning objectives are usually non-convex.

## 6. Experiments and result

The paper evaluates Adam on several machine learning models: logistic regression, multilayer neural networks, convolutional neural networks, and variational autoencoders.

### Logistic Regression

The first experiment uses logistic regression on MNIST and IMDB bag-of-words features.

On MNIST, Adam converges similarly to SGD with Nesterov momentum and faster than AdaGrad.

On the sparse IMDB bag-of-words dataset, Adam performs similarly to AdaGrad and significantly better than SGD with momentum. This shows that Adam can handle sparse gradients effectively.

### Multi-layer Neural Networks

The paper trains multilayer neural networks on MNIST with two fully connected hidden layers of 1000 ReLU units.

With dropout regularization, Adam converges faster than AdaGrad, RMSProp, SGD with Nesterov momentum, and AdaDelta.

For deterministic objectives, Adam is also compared with SFO. Adam makes faster progress in both iterations and wall-clock time, while SFO requires more memory and is slower per iteration.

### Convolutional Neural Networks

The CNN experiment is conducted on CIFAR-10.

Adam and AdaGrad reduce the training cost quickly in the early stage. However, over longer training, Adam and SGD with momentum converge much faster than AdaGrad.

This suggests that Adam avoids the overly aggressive learning-rate decay problem of AdaGrad while still adapting the learning rate for different parameters and layers.

### Bias Correction

The paper also studies the effect of bias correction using a variational autoencoder.

When the bias correction terms are removed, training becomes unstable, especially when $\beta_2$ is close to 1.

This supports the importance of the bias-corrected estimates:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

### Summary

Overall, Adam performs robustly across convex and non-convex problems, dense and sparse gradients, and noisy stochastic objectives. The experiments support the claim that Adam combines the benefits of momentum-based methods and adaptive learning-rate methods such as AdaGrad and RMSProp.

## 7. Thinkings

Adam can be understood as an optimizer that combines the advantages of AdaGrad and RMSProp, while adding bias correction to handle the zero-initialization problem of moment estimates.

Unlike vanilla SGD, Adam does not directly update parameters using the current gradient alone. It uses the moving average of gradients to estimate the update direction and the moving average of squared gradients to adapt the scale of each parameter update.

The convergence analysis strengthens the theoretical motivation of Adam by proving a regret bound in the online convex optimization setting. However, this guarantee does not directly imply convergence for general non-convex deep neural networks.

One question I had is whether the decay rates $\beta_1$ and $\beta_2$ could be learned instead of being fixed hyperparameters. This would turn part of the optimizer design into a learnable process, but it may introduce additional instability and computational overhead.

I initially thought of $\epsilon$ as a local learning-rate parameter, but it is more accurately a numerical stability term. The parameter-wise adaptive step size mainly comes from dividing the global learning rate $\alpha$ by $\sqrt{\hat{v}_t}+\epsilon$.

## 8. Connection to My Research
