# multi-label-classification
 implement and combine an image classifier containing recurrent neural networks (RNNs) (LSTM and GRU) for caption-label analy- sis with EfficientNet convolution networks for multi- label image classification.

# competetion ranking 8th out of 124
https://www.kaggle.com/c/2020s1comp5329assignment2/overview

# Data (find at kaggle pages)
train.csv - the training set
test.csv - the test set
data/*.jpg - the images

# Run the project

Environment
torch 1.5.0, torchvision 0.6.0, efficient-pythorch 0.6.3, dumpy 1.18.4, pandas 1.0.3, tensor flow 1.13.1, torch summary 1.5.1, sklearn 0.0, 

0. Directory
Extract the python file from Algorithm, and put the in the same directory with input and output folder

1. Image input
split images into test image and train image, separately putting them under the ”data/image/test image” and ”data/image/train image”. 

2. Training
Go to train efficientnet and run the python file, the output model will be stored under the Weights direc- tory. 

3. Prediction
Run inference.py, the result will be stored in output folder. 

