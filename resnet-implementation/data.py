import torch.nn
import torch
import torch.nn.functional
import torchvision


def data_loading_augmentation():
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.TenCrop(size=224, vertical_flip=False),
        torchvision.transforms.Lambda(lambda crops: torch.stack(
            [torchvision.PILToTensor()(crop) for crop in crops]))
        # TODO: PCA (https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html)?
    ])

    PATH_TO_IMAGENET_TRAIN = ""
    PATH_TO_IMAGENET_VAL = ""

    imagenet_train = torchvision.datasets.ImageNet(
        root=PATH_TO_IMAGENET_TRAIN, split='train', transform=transforms, download=True)
    imagenet_test = torchvision.datasets.ImageNet(
        root=PATH_TO_IMAGENET_VAL, split='val', transform=transforms, download=True)
    train_loader = torch.utils.data.DataLoader(
        imagenet_train, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        imagenet_test, batch_size=256, shuffle=False)

    return imagenet_train, imagenet_test, train_loader, test_loader
