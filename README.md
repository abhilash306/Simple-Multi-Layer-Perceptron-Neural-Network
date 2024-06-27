# Introduction To Machine Learning
## ELL784

### Assignment 3
#### Neural Network

Submitted to Sumantra Dutta Roy

Pradyot (2022JTM2401)

Abhilash (2022JTM2391)

---

## Index
- Introduction
- Dataset Explanation
- Model
- Model Results
- Effect of Parameter Variation

---

## Introduction

The assignment consists of two parts.

### Part 1
Part 1 of the assignment focuses on designing a simple MLP-based network for 10-class classification. We have to experiment with different network architectures, varying the number of layers and neurons in each layer, as well as training parameters such as learning rate and number of iterations. Interpretability is emphasized, where participants are required to analyze the features learned by the best model. This involves visualizing intermediate representations of different layers and making interpretations from these representations. Participants will report on both correctly classified and misclassified examples, analyzing the features the model is able to capture and those it struggles with. Conclusions will address the shortcomings of the model and propose methods to overcome them. Additionally, participants will consider whether every training example should be given equal weightage and suggest approaches to address this issue.

### Part 2
Part 2 of the assignment challenges us to improve digit classification performance on the dataset through the design of a custom model. The task is open-ended, encouraging participants to delve into the intricacies of machine learning research by developing a unique architecture tailored specifically for digit image classification. Participants are tasked with incorporating mechanisms to capture both local and global details present within the images. This entails designing a model that can effectively encode and learn from small neighbourhoods of pixels (local details) while also correlating these local features to capture broader patterns and structures across the entire image (global details). The overarching goal is to create a unified architecture that seamlessly integrates mechanisms for capturing both local and global features, with a focus on ensuring adequate normalization techniques are employed to enhance the utilization of these features within the model.

---

## Dataset Explanation
We have worked on the MNIST dataset, a well-known dataset for digit classification which contains handwritten digits from 0-9. The MNIST dataset provides separate Train and Test sets. The training dataset has 60000 handwritten digits from 0-9 and the test data has 10000 handwritten digits from 0-9. The sample of handwritten digits is given below:

---

## Model
The neural network model for 10-class classification contains the input layer, hidden layers, and output layers. The first input layer is a flatten layer which flattens the input data (in our case input data is 28x28, after flattening it is 784). We have added seven hidden layers with a decreasing number of neurons in each layer except the batch normalization layer. In the first hidden layer, we have 512 neurons, then 256 neurons, and so on. The final output layer has 10 neurons as output. Total parameters are 1,731,200, out of which 577,002 are trainable parameters. In the hidden layers, we used 'Relu' as the activation function and 'Softmax' in the output layers. For optimization, we used 'Adam optimizer' in our neural network. For the loss function, we used 'categorical' to calculate the loss at each stage.

After training the neural network with a 0.001 learning rate and 15 epochs, we achieved the highest accuracy of 98.42%.

The following figure shows the training loss vs validation loss during training (we split the training data as 0.1% as validation data):

![Training Loss vs Validation Loss](![Capture](https://github.com/abhilash306/Simple-Multi-Layer-Perceptron-Neural-Network/assets/29005113/16f5c48d-2f3f-49eb-8c46-a378df88a315)
)

The following figure shows the training accuracy vs validation accuracy during training (we split the training data as 0.1% as validation data):

![Training Accuracy vs Validation Accuracy](![Capture1](https://github.com/abhilash306/Simple-Multi-Layer-Perceptron-Neural-Network/assets/29005113/7d1722fe-0594-4852-bd3a-334086f6c6b5)


The confusion matrix for 10-class classification is given below:

![Confusion Matrix](link-to-image)

We experimented with different parameters like the number of hidden layers, the number of neurons in each layer, learning rate, and number of iterations. The table below shows the accuracy of different experiments. The color coding shows the variation of parameters and their respective accuracy.

| Number of Hidden Layers | Number of Neurons in Each Layer | Learning Rate | Number of Iterations | Accuracy |
|-------------------------|---------------------------------|---------------|----------------------|----------|
| 5                       | 512, 256, 128, 64, 32           | 0.001         | 15                   | 98.42%   |
| 4                       | 256, 128, 64, 32                | 0.001         | 15                   | 97.99%   |
| 3                       | 128, 64, 32                     | 0.001         | 15                   | 97.53%   |
| 5                       | 512, 512, 128, 64, 32           | 0.001         | 15                   | 97.97%   |
| 5                       | 256, 256, 256, 256, 256         | 0.001         | 15                   | 97.64%   |
| 5                       | 32, 32, 32, 32, 32              | 0.001         | 15                   | 96.79%   |
| 5                       | 512, 256, 128, 64, 32           | 0.1           | 15                   | 97.13%   |
| 5                       | 512, 256, 128, 64, 32           | 0.01          | 15                   | 97.82%   |
| 5                       | 512, 256, 128, 64, 32           | 0.0001        | 15                   | 97.57%   |
| 5                       | 512, 256, 128, 64, 32           | 0.001         | 5                    | 97.08%   |
| 5                       | 512, 256, 128, 64, 32           | 0.001         | 10                   | 98.089%  |
| 5                       | 512, 256, 128, 64, 32           | 0.001         | 25                   | 98.11%   |

---

## Interpretability
To understand which features are learned and extracted by the neural network, we used a filter visualization technique. In this technique, for the dense layers, we can inspect the weights to see what features each neuron is learning. Following are some pictures which show the filter at different layers of our neural network:

![1st Hidden Layer Filter](![Capture4](https://github.com/abhilash306/Simple-Multi-Layer-Perceptron-Neural-Network/assets/29005113/f0400011-eeee-4090-8311-49bad33a116e)
)

![2nd Hidden Layer Filter](![Capture5](https://github.com/abhilash306/Simple-Multi-Layer-Perceptron-Neural-Network/assets/29005113/726c347a-7ebc-495d-a867-d0b65a782e6a)
)
---

## Analyzing the Misclassification
In the case of misclassification, generally what we observe is that the neural network is not giving good results where one digit is somehow similar to another digit; in that case, they give the wrong prediction. One way to solve this problem is using local and global detailing, where the neural network will extract only the local details of the digit based on which they will give the prediction about the digit.

---

## Part 2

### Local Feature Extraction
The local feature extraction stage involves applying convolution operations followed by an activation function, and optionally batch normalization and pooling.

Let \(X\) represent the input image of size \(W \times H \times C\), where \(W\) is the width, \(H\) is the height, and \(C\) is the number of channels.

#### Convolution Operation
Let \(W_{\text{local}}\) be the filter weights for the local feature extraction layer. The output feature map \(f_{\text{local}}\) is computed as:
\[ f_{\text{local}} = \text{ReLU}(X * W_{\text{local}} + b_{\text{local}}) \]
where \(*\) denotes the convolution operation and \(b_{\text{local}}\) is the bias term.

#### Pooling Operation
Let \(P_{\text{local}}\) represent the pooling operation. The output feature map after pooling \(f'_{\text{local}}\) is computed as:
\[ f'_{\text{local}} = P_{\text{local}}(f_{\text{local}}) \]

### Global Feature Extraction
Similar to local feature extraction, the global feature extraction stage involves convolution, activation, optional normalization, and pooling operations.

Let \(f'_{\text{local}}\) represent the output feature map from the local feature extraction stage.

#### Convolution Operation
Let \(W_{\text{global}}\) be the filter weights for the global feature extraction layer. The output feature map \(f_{\text{global}}\) is computed as:
\[ f_{\text{global}} = \text{ReLU}(f'_{\text{local}} * W_{\text{global}} + b_{\text{global}}) \]

#### Pooling Operation
Let \(P_{\text{global}}\) represent the pooling operation. The output feature map after pooling \(f'_{\text{global}}\) is computed as:
\[ f'_{\text{global}} = P_{\text{global}}(f_{\text{global}}) \]

### Fusion of Local and Global Features
The feature maps obtained from the local and global feature extraction stages are concatenated. Let \(f_{\text{concat}}\) represent the concatenated feature map. The concatenation operation is performed along the channel dimension:
\[ f_{\text{concat}} = [f_{\text{local}}, f'_{\text{global}}] \]

### Fully Connected Layers
The concatenated feature map is flattened and passed through fully
