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

### Training on Multiple GPU

The AlexNet model was too large to fit into the memory of a single GTX 580 GPU, which had only 3GB of memory. Therefore, the authors split the network across two GPUs.

The main idea is to place roughly half of the kernels or neurons on each GPU. However, the two GPUs do not communicate at every layer. Some layers receive inputs from feature maps on both GPUs, while other layers only receive inputs from the feature maps located on the same GPU.

This design reduces the memory burden and allows the model to be trained efficiently, while controlling the communication cost between GPUs. In the paper, this two-GPU scheme also improves top-1 and top-5 error rates compared to a smaller one-GPU network.

### Local Response Normalization

Local Response Normalization normalizes the activation of a neuron using the activations of nearby feature maps at the same spatial position.

```math
b_{x,y}^{i}
=
\frac{
a_{x,y}^{i}
}{
\left(
k + \alpha
\sum_{j=\max(0,\, i-n/2)}^{\min(N-1,\, i+n/2)}
\left(a_{x,y}^{j}\right)^2
\right)^{\beta}
}
```

Here, $a_{x,y}^{i}$ is the activation before normalization, and $b_{x,y}^{i}$ is the normalized activation. The summation is computed over nearby feature maps at the same spatial position $(x,y)$.

The intuition is that if many nearby feature maps have large activations at the same location, the denominator becomes large and suppresses the current activation. This creates a competition between feature maps, similar to lateral inhibition.

In AlexNet, this normalization was used after ReLU in certain layers and helped improve generalization.
### Overlapping Pooling

pooling window size: z × z
stride: s

then standard pooling s=z, overlapping pooling s<z. This paper found overlapping pooling help generalization.

### Architecture 

AlexNet consists of eight learned layers: five convolutional layers and three fully-connected layers.

The overall flow is:

```text
Input image
→ Conv1 → ReLU → Local Response Normalization → Max Pooling
→ Conv2 → ReLU → Local Response Normalization → Max Pooling
→ Conv3 → ReLU
→ Conv4 → ReLU
→ Conv5 → ReLU → Max Pooling
→ FC6 → ReLU
→ FC7 → ReLU
→ FC8 → Softmax
```
The convolutional layers extract visual features from the input image, while the fully-connected layers use these features for final classification. The last fully-connected layer produces 1000 outputs, corresponding to the 1000 ImageNet classes.

A notable feature of AlexNet is that the model is split across two GPUs. Some convolutional layers communicate across both GPUs, while others only use feature maps from the same GPU. This design reduces memory and communication costs while allowing a large CNN to be trained efficiently.

## 5. Reducing Overfitting

### Data Augmentation

Data augmentation is a technique for reducing overfitting by artificially enlarging the training dataset using label-preserving transformations.

In AlexNet, the authors used two main forms of data augmentation.

First, they generated image translations and horizontal reflections. The original images were resized to $256 \times 256$, and random $224 \times 224$ patches were extracted during training. Horizontal reflections of these patches were also used. This allowed the network to see many different versions of the same image while keeping the label unchanged.

At test time, the network used five fixed $224 \times 224$ crops: the four corner patches and the center patch. It also used their horizontal reflections, resulting in ten patches in total. The final prediction was obtained by averaging the softmax outputs over these ten patches.

Second, the authors altered the intensities of the RGB channels. They performed PCA on the RGB pixel values of the ImageNet training set and added small random perturbations along the principal component directions. This augmentation makes the model more robust to changes in lighting and color.

Overall, data augmentation helps prevent the model from memorizing the training images and improves generalization by making the model robust to translation, reflection, and illumination changes.

### Dropout

Dropout is a regularization technique that randomly sets the output of each hidden neuron to zero during training.

In AlexNet, each hidden neuron is dropped with probability $0.5$.

During a training iteration, dropped neurons do not contribute to the forward pass and do not participate in backpropagation. However, the remaining active neurons are still trained normally by backpropagation.

This means that each training iteration samples a different subnetwork, while all subnetworks share the same weights. As a result, neurons cannot rely too much on specific other neurons, which reduces co-adaptation and encourages the network to learn more robust features.

At test time, all neurons are used, but their outputs are multiplied by $0.5$ to match the expected activation scale during training.

In AlexNet, dropout was applied to the first two fully-connected layers. Without dropout, the network showed substantial overfitting.

## 6. Experiments and Results



## 7. Thinkings 

How about using batch norm in Alexnet? we need to re-tunning learning rate, weight decay, dropout , initialization
batch size, optimizer, training schedule but it seems valid to use batch norm.

## 8. Connection to My Research
