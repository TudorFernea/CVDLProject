import torch
import torchvision
from torchvision import transforms
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import resNet
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from alexNet import Alexnet

#os.chdir('C:/Users/tudor/Desktop/CVDL')

torch.cuda.empty_cache()
print(torch.cuda.get_device_name(0))
data_transforms = transforms.Compose([
    transforms.Resize([112, 112]),
    transforms.ToTensor()
])

BATCH_SIZE = 128
learning_rate = 0.001
EPOCHS = 15
numClasses = 43


train_data_path = "./GTSRB/Train"
train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=data_transforms)

ratio = 0.8
n_train_examples = int(len(train_data) * ratio)
n_val_examples = len(train_data) - n_train_examples

train_data, val_data = data.random_split(train_data, [n_train_examples, n_val_examples])

print(f"Number of training samples = {len(train_data)}")
print(f"Number of validation samples = {len(val_data)}")

train_hist = [0] * numClasses
for i in train_data.indices:
    tar = train_data.dataset.targets[i]
    train_hist[tar] += 1

val_hist = [0] * numClasses
for i in val_data.indices:
    tar = val_data.dataset.targets[i]
    val_hist[tar] += 1

plt.bar(range(numClasses), train_hist, label="train")
plt.bar(range(numClasses), val_hist, label="val")
legend = plt.legend(loc='upper right', shadow=True)
plt.title("Distribution Plot")
plt.xlabel("Class ID")
plt.ylabel("# of examples")

plt.savefig("train_val_split.png", bbox_inches='tight', pad_inches=0.5)


train_loader = data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
val_loader = data.DataLoader(val_data, shuffle=True, batch_size=BATCH_SIZE)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



#model = Alexnet(numClasses) #AlexNet
model = restNet.create_resNet(numClasses) #ResNet
print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    print(torch.cuda.current_device())
    model = model.cuda()
    criterion = criterion.cuda()

print(model)

print(summary(model, (3, 112, 112)))

print("Model's state dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
print("")

print("Optimizer details:")
print(optimizer)
print("")

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train(model, loader, opt, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (images, labels) in loader:
        images = images.cuda()
        labels = labels.cuda()

        opt.zero_grad()

        #output, _ = model(images) #AlexNet
        output = model(images) #ResNet
        loss = criterion(output, labels)

        loss.backward()

        acc = calculate_accuracy(output, labels)

        opt.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)



def evaluate(model, loader, opt, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (images, labels) in loader:
            images = images.cuda()
            labels = labels.cuda()

#           output, _ = model(images) #AlexNet
            output = model(images) #ResNet

            loss = criterion(output, labels)

            acc = calculate_accuracy(output, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)


train_loss_list = [0] * EPOCHS
train_acc_list = [0] * EPOCHS
val_loss_list = [0] * EPOCHS
val_acc_list = [0] * EPOCHS

for epoch in range(EPOCHS):
    print("Epoch-%d: " % (epoch))

    train_start_time = time.monotonic()
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    train_end_time = time.monotonic()

    val_start_time = time.monotonic()
    val_loss, val_acc = evaluate(model, val_loader, optimizer, criterion)
    val_end_time = time.monotonic()

    train_loss_list[epoch] = train_loss
    train_acc_list[epoch] = train_acc
    val_loss_list[epoch] = val_loss
    val_acc_list[epoch] = val_acc

    print("Training: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds" % (
    train_loss, train_acc, train_end_time - train_start_time))
    print("Validation: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds" % (
    val_loss, val_acc, val_end_time - val_start_time))
    print("")

MODEL_FOLDER = "./Model"
if not os.path.isdir(MODEL_FOLDER):
    os.mkdir(MODEL_FOLDER)

#PATH_TO_MODEL = MODEL_FOLDER + "/pytorch_classification_alexnetTS.pth" #AlexNet
PATH_TO_MODEL = MODEL_FOLDER + "/pytorch_classification_ResNet.pth" #ResNet

if os.path.exists(PATH_TO_MODEL):
    os.remove(PATH_TO_MODEL)
torch.save(model.state_dict(), PATH_TO_MODEL)

print("Model saved at %s" % (PATH_TO_MODEL))

_, axs = plt.subplots(1, 2, figsize=(15, 5))

axs[0].plot(train_loss_list, label="train")
axs[0].plot(val_loss_list, label="val")
axs[0].set_title("Plot - Loss")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
legend = axs[0].legend(loc='upper right', shadow=False)

axs[1].plot(train_acc_list, label="train")
axs[1].plot(val_acc_list, label="val")
axs[1].set_title("Plot - Accuracy")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Accuracy")
legend = axs[1].legend(loc='center right', shadow=True)

plt.savefig("train_val_epoch.png", bbox_inches='tight', pad_inches=0.5)
