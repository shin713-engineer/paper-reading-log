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

Algorithm: 
1. gradient messure

## 6. Important Equations

gradient : \[g_t = \nabla_\theta f_t(\theta_{t-1})\]

momentum
mt​=β1​⋅mt−1​+(1−β1​)⋅gt​

vt​=β2​⋅vt−1​+(1−β2​)⋅gt2​

## 7. Experiments and Results
## 8. Strengths
## 9. Limitations
## 10. Connection to My Research
