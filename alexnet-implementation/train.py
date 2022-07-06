import torch.nn
import torch
import torch.nn.functional
from tqdm.notebook import tqdm, trange

from model import AlexNet
from data import data_loading_augmentation

#### data loading and augmentation ####

imagenet_train, imagenet_test, train_loader, test_loader = data_loading_augmentation()

#### training ####

net = AlexNet()

criterion = torch.nn.CrossEntropyLoss(
    weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean', label_smoothing=0.0)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9,
#                             dampening=0, weight_decay=0.0005, nesterov=False, maximize=False)
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.01)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1)

for epoch in trange(90):
    lr_scheduler.step()
    for images, labels in tqdm(train_loader):
        optimizer.zero_grad()

        # Forward pass
        x = images
        y = net(x)
        loss = criterion(y, labels)
        # Backward pass
        loss.backward()
        optimizer.step()

#### testing ####

correct = 0
total = len(imagenet_test)

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        x = images
        y = net(x)

        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())

print('Test accuracy: {}'.format(correct/total))

torch.save(net.state_dict(), "model.pth")
