import torch
from tqdm.notebook import trange


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(64 * 8),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 4),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 2),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)


class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64 * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


train, test, train_loader, test_loader = None, None, None, None

gen = Generator()
disc = Discriminator()

criterion = torch.nn.BCELoss()

gen_optimizer = torch.optim.Adam(params=gen.parameters(), lr=0.01)
disc_optimizer = torch.optim.Adam(params=disc.parameters(), lr=0.01)

for epoch in trange(90):
    for i, (imgs, _) in enumerate(train_loader):
        valid = torch.autograd.Variable(torch.FloatTensor(
            imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = torch.autograd.Variable(torch.FloatTensor(
            imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        real_imgs = torch.autograd.Variable(imgs.type(torch.FloatTensor))

        # generator

        gen_optimizer.zero_grad()
        noise = torch.randn(imgs.shape, 100, 1, 1, device='cpu')
        gen_imgs = gen(noise)
        gen_loss = criterion(disc(gen_imgs), valid)
        gen_loss.backward()
        gen_optimizer.step()

        # discriminator

        disc_optimizer.zero_grad()

        real_loss = criterion(disc(real_imgs), valid)
        fake_loss = criterion(disc(gen_imgs.detach()), fake)
        disc_loss = (real_loss + fake_loss) / 2

        disc_loss.backward()
        disc_optimizer.step()
