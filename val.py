import torch
import torchvision
from torchvision import transforms
import torch.utils.data as data
import time
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

import resNet
from alexNet import Alexnet

torch.cuda.empty_cache()
test_transforms = transforms.Compose([
    transforms.Resize([112, 112]),
    transforms.ToTensor()
])


test_data_path = "./GTSRB/Test"
test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transforms)
test_loader = data.DataLoader(test_data, batch_size=1, shuffle=False)
numClasses = 43

num = range(numClasses)
labels = []
for i in num:
    labels.append(str(i))
labels = sorted(labels)
for i in num:
    labels[i] = int(labels[i])
print("List of labels : ")
print("Actual labels \t--> Class in PyTorch")
for i in num:
    print("\t%d \t--> \t%d" % (labels[i], i))

df = pd.read_csv("./GTSRB/Test2.csv")
numExamples = len(df)
labels_list = list(df.ClassId)


#MODEL_PATH = "./Model/pytorch_classification_alexnetTS.pth"
MODEL_PATH = "./Model/pytorch_classification_ResNet.pth"

#model = Alexnet(numClasses)
model = restNet.create_resNet(numClasses)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.cuda()


y_pred_list = []
corr_classified = 0

with torch.no_grad():
    model.eval()

    i = 0

    for image, _ in test_loader:

        image = image.cuda()

        #img = image.cpu().numpy()
        # transpose image to fit plt input
        #img = img.T
        #plt.imshow(img[:,:,0,0])
        #plt.show()

        y_test_pred = model(image)

        #y_pred_softmax = torch.log_softmax(y_test_pred[0], dim=1) #AlexNet
        y_pred_softmax = y_test_pred #ResNet
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
        y_pred_tags = y_pred_tags.cpu().numpy()

        y_pred = y_pred_tags[0]
        y_pred = labels[y_pred]

        y_pred_list.append(y_pred)

        if labels_list[i] == y_pred:
            corr_classified += 1


        i += 1

print("Number of correctly classified images = %d" % corr_classified)
print("Number of incorrectly classified images = %d" % (numExamples - corr_classified))
print("Final accuracy = %f" % (corr_classified / numExamples))


print(classification_report(labels_list, y_pred_list))


def plot_confusion_matrix(labels, pred_labels, classes):
    fig = plt.figure(figsize=(20, 20));
    ax = fig.add_subplot(1, 1, 1);
    cm = confusion_matrix(labels, pred_labels);
    cm = ConfusionMatrixDisplay(cm, display_labels=classes);
    cm.plot(values_format='d', cmap='Blues', ax=ax)
    plt.xticks(rotation=20)


labels_arr = range(0, numClasses)
plot_confusion_matrix(labels_list, y_pred_list, labels_arr)
print(y_pred_list[:20])
print(labels_list[:20])

