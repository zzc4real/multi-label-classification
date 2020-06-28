# multi-label-classification
 implement and combine an im- age classifier containing recurrent neural networks (RNNs) (LSTM and GRU) for caption-label analy- sis with EfficientNet convolution networks for multi- label image classification.

# competetion ranking 8th out of 124
https://www.kaggle.com/c/2020s1comp5329assignment2/overview

# Data (find at kaggle pages)
train.csv - the training set
test.csv - the test set
data/*.jpg - the images

# Run the project
To start with, split images into test im- age and train image, separately putting them under the ”data/image/test image” and ”data/image/train image”. For the train process, go to train efficientnet and run the python file, the output model will be stored under the Weights direc- tory. In the process of prediction, run inference.py, the result will be stored in output folder.
