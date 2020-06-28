import numpy as np
import pandas as pd
import data_preprocess_cnn
import os
import PIL
import sys
import torch
from time import time
import torchvision
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from torchsummary import summary
import tensorflow as tf

class Dataset(data.Dataset):
    def __init__(self, csv_path, images_path, transform=None):
        self.csv_path = csv_path
        self.train_set = pd.read_csv(csv_path)['ImageID']  # Read The CSV and create the dataframe
        self.train_path = images_path  # Images Path
        self.transform = transform  # Augmentation Transforms

    def __len__(self):
        return len(self.train_set)

    def __getitem__(self, idx):
        file_name = self.train_set.iloc[idx][0] + '.jpg'
        label = data_preprocess_cnn.load_data(self.csv_path)[idx]
        img = Image.open(os.path.join(self.train_path, file_name))  # Loading Image
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def main():
    # load image files name
    # for dirname, _, filenames in os.walk('data/image/train_image'):
    #     for filename in filenames:
    #         print(os.path.join(dirname, filename))

    # load csv data set
    BASE_PATH = 'data/'
    train_dataset = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
    test_dataset = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))

    # Defining Transforms and Parameters for Training
    # Network Params
    params = {'batch_size': 16,
              'shuffle': True
              }
    epochs = 5

    learning_rate = 0.01

    # define image transform
    transform_train = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomApply([
        torchvision.transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip()], 0.7),
        transforms.ToTensor()])

    training_set = Dataset(os.path.join(BASE_PATH, 'train.csv'), os.path.join(BASE_PATH, 'image/train_image'),
                           transform=transform_train)

    training_generator = data.DataLoader(training_set, **params)


    # # define cpu usage
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    # print(device)

    # import the model
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=19)
    # model.to(device)
    print(summary(model, input_size=(3, 512, 512)))
    PATH_SAVE = './Weights/'
    if (not os.path.exists(PATH_SAVE)):
        os.mkdir(PATH_SAVE)
    criterion = nn.BCELoss()
    m = nn.Sigmoid()
    lr_decay = 0.99
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train the model

    # Eye to create an 5x5 tensor
    # eye = torch.eye(5).to(device)
    classes = [0, 1, 2, 3, 4]
    history_accuracy = []
    history_loss = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        class_correct = list(0. for _ in classes)
        class_total = list(0. for _ in classes)

        for i, tdata in enumerate(training_generator, 0):

            inputs, labels = tdata
            # print(labels)
            t0 = time()

            # using gpu
            # inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # print(labels)
            # print(torch.max(labels, 1))
            # print(torch.max(labels, 1)[1])
            flabels = labels.float()
            loss = criterion(m(outputs), flabels)
            # print(loss)

            # construct predict tensor
            predicted = m(outputs)
            zero = torch.zeros_like(predicted)
            one = torch.ones_like(predicted)
            predicted = torch.where(predicted>0.5, one, zero)

            batch_correct = 0
            # c = (predicted == labels.data).squeeze()
            for p in range(len(predicted)):
                cont = 0
                for q in range(len(predicted[p])):
                    if predicted[p][q] == flabels[p][q]:
                        cont += 1
                if cont == 19:
                    correct += 1
                    batch_correct += 1

            total += predicted.size(0)
            accuracy = float(correct) / float(total)
            batch_accuracy = float(batch_correct) / float(len(predicted))

            # history_accuracy.append(accuracy)
            history_accuracy.append(accuracy)
            history_loss.append(loss)

            loss.backward()
            optimizer.step()

            # for j in range(labels.size(0)):
            #     label = labels[j]
            #     class_correct[label] += c[j].item()
            #     class_total[label] += 1

            running_loss += loss.item()

            print("Epoch : ", epoch + 1, " Batch : ", i + 1, " Loss :  ", running_loss / (i + 1), " Accumulated Accuracy : ", accuracy, " Batch Accuracy", batch_accuracy, "Time ", round(time() - t0, 2), "s")
        # for k in range(len(classes)):
        #     if (class_total[k] != 0):
        #         print('Accuracy of %5s : %2d %%' % (classes[k], 100 * class_correct[k] / class_total[k]))

        if epoch % 10 == 0 or epoch == 0:
            torch.save(model.state_dict(), os.path.join(PATH_SAVE, str(epoch + 1) + '_' + str(running_loss) + '.pth'))

    torch.save(model.state_dict(), os.path.join(PATH_SAVE, 'Last_epoch' + str(running_loss) + '.pth'))


if __name__ == "__main__":
    main()
