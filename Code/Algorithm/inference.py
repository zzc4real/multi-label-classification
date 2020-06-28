from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
import os
from PIL import Image
import pandas as pd
from torch.autograd import Variable
import torch
import torch.nn as nn
model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=19)

model.load_state_dict(torch.load('Weights/Last_epoch367.79824512451887.pth'))

model.eval()

test_transforms = transforms.Compose([transforms.Resize(512),
                                      transforms.ToTensor(),])
m = nn.Sigmoid()

def predict_image(image):
    image_tensor = test_transforms(image)
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    # input = input.to(device)
    output = model(input)
    predicut = m(output)
    record = []
    print(predicut)
    for i in range(len(predicut[0])):
        if predicut[0][i] > 0.5:
            x = i+1
            record.append(x)


    return record

submission=pd.read_csv('data/test.csv')
submission_csv=pd.DataFrame(columns=['ImageID'])

IMG_TEST_PATH='data/image/test_image/'


for i in range(len(submission)):
    img=Image.open(IMG_TEST_PATH+submission.iloc[i][0])
    prediction=predict_image(img)
    num_prediction = [str(x) for x in prediction]
    s = ' '.join(num_prediction)
    submission_csv = submission_csv.append({'ImageID': submission.iloc[i][0], 'Labels': s},
                                           ignore_index=True)

submission_csv.to_csv('output/submission.csv',index=False)