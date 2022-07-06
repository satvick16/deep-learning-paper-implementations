import torch

from model import AlexNet

PATH = "path to model.pth"

model = AlexNet()
model.load_state_dict(torch.load(PATH))
model.eval()
