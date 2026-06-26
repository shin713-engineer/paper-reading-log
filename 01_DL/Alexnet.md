# Paper Title
## 1. Paper Information
- Title: ImageNet Classification with Deep Convolutional Neural Networks
- Authors: Alex, Krizhevsky
- Year: 2012
- Topic: Deep neural network
- Link: https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
## 2. One-line Summary

Alexnet is CNN which is trained for large training set. 
It has much more strength in large dataset and ensures less possibility for overfitting. 

## 3. Problem

makes CNN more powerful - using GPU, data augmentation, RELU, dropout 
CNN is useful in image classification, object recognition and ALexnet makes it possible for large dataset.
introduce featurs used for improving performance and reducing training time 
alse introduce the method to prevent overfitting

## 4. Method / Architecture

## ReLU 

### ReLU Nonlinearity

Traditional activation functions such as the sigmoid function and hyperbolic tangent function are saturating nonlinearities:

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

$$
f(x) = \tanh(x)
$$

These functions can suffer from the vanishing gradient problem. When the input $x$ is very large or very small, the function becomes almost flat, so its gradient becomes close to zero.

In contrast, ReLU is defined as:

$$
f(x) = \max(0, x)
$$

For positive inputs, the derivative of ReLU is always $1$:

$$
f'(x) = 1 \quad \text{for } x > 0
$$

Therefore, gradients can flow more easily during backpropagation. This is why ReLU allows deep neural networks to train faster than saturating activation functions such as sigmoid or tanh.


## 5. Important Equations
## 6. Experiments and Results
## 7. Thinkings 
## 8. Connection to My Research
