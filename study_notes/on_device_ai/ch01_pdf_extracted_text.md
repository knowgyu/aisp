# Chapter 1 PDF Extracted Text


## Slide 1

Intelligent Embedded Systems Lab. @ SKKU
28
1
Dongkun Shin
Intelligent Embedded Systems Lab.
Sungkyunkwan University
On-Device AI 실습
Pruning for CNN


## Slide 2

Intelligent Embedded Systems Lab. @ SKKU
28
2
•
Download practice materials
–
https://drive.google.com/drive/folders/1GZRDpshMbKn6f46Wdoj204tXMi4dOrcQ?usp=dri
ve_link
–
Short Link: https://bit.ly/4fQXk3f
•
Open a command prompt or PowerShell in the directory containing the materials, set up 
the conda environment, and run Jupyter Lab
•
If you encounter environment issues, download the provided .bat file to perform a 
rollback
–
\\swschoolavdazfiles002.file.core.windows.net\aias-language\Env\Rollback_env_aias_test.bat
Practice Setup


## Slide 3

Intelligent Embedded Systems Lab. @ SKKU
28
3
1.
Pruning Granularity
–
Learn and implement the differences between Fine-grained, Vector-level, 
Kernel-level, and Channel-level pruning
2.
Pruning Ratio
–
Learn how to determine the sparsity (ratio of weights to remove) and check 
the difference between Layer-wise and Global methods through practice
3.
Pruning Schedule
–
Understand One-shot and Iterative Pruning, and apply Linear and Cubic 
Scheduling to analyze performance changes
Overview


## Slide 4

Intelligent Embedded Systems Lab. @ SKKU
28
4
•
A dataset is required to train a model; this 
practice uses the CIFAR-10 dataset
•
In PyTorch, datasets are managed using a 
Python class called DataLoader
–
It configures pre-processing (transforms), 
batch size, shuffling, and the number of 
workers
•
CIFAR-10 consists of 32x32 color images across 
10 classes
–
Train: 50,000
–
Test: 10,000
PyTorch Basic Tutorial: Dataset
https://www.cs.toronto.edu/~kriz/cifar.html
CIFAR-10 Dataset


## Slide 5

Intelligent Embedded Systems Lab. @ SKKU
28
5
PyTorch Basic Tutorial: Dataset (Code snippet)
Data 
augmentation


## Slide 6

Intelligent Embedded Systems Lab. @ SKKU
28
6
PyTorch Basic Tutorial: Model
In this practice, a pre-trained model is loaded using torch.hub


## Slide 7

Intelligent Embedded Systems Lab. @ SKKU
28
7
•
In PyTorch, all models and layers 
are defined via the nn.Module
class and organized hierarchically
•
The practice uses a VGG9 model 
pre-trained on CIFAR-10
–
consisting of 8 convolution 
layers and 1 linear layer
PyTorch Basic Tutorial: Model Structure
VGG(
(backbone): Sequential(
(conv0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu0): ReLU(inplace=True)
(conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu1): ReLU(inplace=True)
(pool0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(conv2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu2): ReLU(inplace=True)
(conv3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu3): ReLU(inplace=True)
(pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(conv4): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu4): ReLU(inplace=True)
(conv5): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn5): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu5): ReLU(inplace=True)
(pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(conv6): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn6): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu6): ReLU(inplace=True)
(conv7): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
(bn7): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(relu7): ReLU(inplace=True)
(pool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
(classifier): Linear(in_features=512, out_features=10, bias=True)
)
VGG9 Model Structure


## Slide 8

Intelligent Embedded Systems Lab. @ SKKU
28
8
PyTorch Basic Tutorial: Evaluating Model Size
count the total number of 
elements that is not 0
count the total number of 
elements
iterating through all parameters in the model


## Slide 9

Intelligent Embedded Systems Lab. @ SKKU
28
9
PyTorch Basic Tutorial: Evaluating Model Accuracy
data is obtained via the 
DataLoader
change the model to evaluation mode
output represents 
confidence values for the 
10 classes, and the index of 
the largest value becomes 
the predicted class
output shape is (batch_size, num_classes)


## Slide 10

Intelligent Embedded Systems Lab. @ SKKU
28
10
PyTorch Basic Tutorial: Parameters
Layer
Parameters
Shape
torch.nn.Conv2d
weight
𝑜𝑢𝑡_𝑐ℎ𝑎𝑛𝑛𝑒𝑙𝑠, 𝑖𝑛_𝑐ℎ𝑎𝑛𝑛𝑒𝑙𝑠
𝑔𝑟𝑜𝑢𝑝𝑠
, 𝑘𝑒𝑟𝑛𝑒𝑙_ℎ𝑒𝑖𝑔ℎ𝑡, 𝑘𝑒𝑟𝑛𝑒𝑙_𝑤𝑖𝑑𝑡ℎ
bias
𝑜𝑢𝑡_𝑐ℎ𝑎𝑛𝑛𝑒𝑙𝑠
torch.nn.Linear
weight
𝑜𝑢𝑡_𝑓𝑒𝑎𝑡𝑢𝑟𝑒𝑠, 𝑖𝑛_𝑓𝑒𝑎𝑡𝑢𝑟𝑒𝑠
bias
𝑜𝑢𝑡_𝑓𝑒𝑎𝑡𝑢𝑟𝑒𝑠
torch.nn.BatchNorm2d
weight
𝑛𝑢𝑚_𝑓𝑒𝑎𝑡𝑢𝑟𝑒𝑠
bias
𝑛𝑢𝑚_𝑓𝑒𝑎𝑡𝑢𝑟𝑒𝑠
E.g., Conv2d와 Linear의 weight를 순회:
or


## Slide 11

Intelligent Embedded Systems Lab. @ SKKU
28
11
Check the weight distribution per layer of the pre-trained model
Many weights are 
distributed close to 0 
➔weights can be 
compressed by pruning


## Slide 12

Intelligent Embedded Systems Lab. @ SKKU
28
12
•
Write pruning code considering various pruning granularities
–
Irregular patterns have a smaller drop in accuracy but lower hardware 
efficiency
–
Regular patterns cause a larger accuracy drop but offer high hardware 
efficiency
1.1. Pruning Granularity/Pattern
Easy to 
Accelerate
Higher 
Accuracy
Input channels
Output channels


## Slide 13

Intelligent Embedded Systems Lab. @ SKKU
28
13
•
Right figure shows the magnitude of a (128, 64, 
3, 3) weight
–
The layout of convolution weights in PyTorch
is (CO, CI, KH, KW)
•
CO: output channels
•
CI: input channels
•
KH: kernel height
•
KW: kernel width
–
For visualization, the layout was reshaped to 
(CO * KH, CI * KW) to arrange the data as 
shown in the previous slide
Print Weight Distribution


## Slide 14

Intelligent Embedded Systems Lab. @ SKKU
28
14
[실습1] Fine-grained Pruning 구현


## Slide 15

Intelligent Embedded Systems Lab. @ SKKU
28
15
[실습1] Answer


## Slide 16

Intelligent Embedded Systems Lab. @ SKKU
28
16
Compares the Importance Sum and Reconstruction Error across 
different pruning methods


## Slide 17

Intelligent Embedded Systems Lab. @ SKKU
28
17
•
Sensitivity Analysis
•
Global Magnitude Pruning
1.2. Pruning Ratio
Because each layer has a different sensitivity, 
different pruning ratios are required for each layer
Layer-wise
Global
A simple global pruning method that performs 
pruning using a single global threshold


## Slide 18

Intelligent Embedded Systems Lab. @ SKKU
28
18
•
Analyzes the sensitivity of the pre-trained model by plotting validation 
accuracy versus pruning sparsity for various layers
Sensitivity analysis


## Slide 19

Intelligent Embedded Systems Lab. @ SKKU
28
19
•
Based on the previous sensitivity 
analysis, assign an appropriate 
sparsity to each layer and evaluate
•
Compare the accuracy with layer-
wise pruning that uses uniform 
sparsity
[실습 2] Sensitivity analysis를 통한 pruning 수행


## Slide 20

Intelligent Embedded Systems Lab. @ SKKU
28
20
[실습2] Answer


## Slide 21

Intelligent Embedded Systems Lab. @ SKKU
28
21
•
Implement global magnitude pruning, 
which prunes using a single global 
threshold
•
Compare the accuracy with layer-wise 
pruning and the global pruning 
derived from the sensitivity analysis
[실습 3] Global magnitude pruning 구현


## Slide 22

Intelligent Embedded Systems Lab. @ SKKU
28
22
[실습3] Answer


## Slide 23

Intelligent Embedded Systems Lab. @ SKKU
28
23
•
One-shot Pruning
•
Iterative Pruning
– Sparsity scheduling
1.3. Pruning Schedule: Overview
One-shot
Iterative


## Slide 24

Intelligent Embedded Systems Lab. @ SKKU
28
24
•
Criterion:
– Loss function
•
Optimizer:
– Algorithm (e.g., SGD) 
that updates model 
parameters to advance 
training
•
Scheduler:
– Learning rate scheduler
Model Training Function
initializing accumulated gradients to 0
calculating loss and gradients via 
backward propagation
updating parameters and learning rates


## Slide 25

Intelligent Embedded Systems Lab. @ SKKU
28
25
One-shot pruning
Training accuracy may vary depending on the 
reading order of the training dataset and the 
pre-processing (augmentation) methods used


## Slide 26

Intelligent Embedded Systems Lab. @ SKKU
28
26
•
Sparsity scheduler
– 𝑣𝑡= 𝑣𝑓+ 𝑣𝑖−𝑣𝑓
1 −
𝑡−𝑡𝑖
𝑡𝑓−𝑡𝑖
𝐸
• E=1: linear
• E=3: cubic
[실습 4] Sparsity scheduler 구현


## Slide 27

Intelligent Embedded Systems Lab. @ SKKU
28
27
[실습4] Answer


## Slide 28

Intelligent Embedded Systems Lab. @ SKKU
28
28
Iterative pruning
Linear Sparsity Scheduler
Cubic Sparsity Scheduler
Training accuracy may vary depending on the 
reading order of the training dataset and the 
pre-processing methods
