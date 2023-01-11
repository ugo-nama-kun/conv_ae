import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn

import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("selected device: ", device)

data_dir = "dataset"

train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)
in_channel = 1


# train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
# test_dataset = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)
# in_channel = 3

# print(test_dataset.targets)

# from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]


if in_channel == 1:
    contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomResizedCrop(size=32),
                                              transforms.GaussianBlur(kernel_size=9),
                                              transforms.Normalize((0.5,), (0.5,))
                                              ])
else:
    contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomResizedCrop(size=32),
                                              transforms.RandomApply([
                                                  transforms.ColorJitter(brightness=0.5,
                                                                         contrast=0.5,
                                                                         saturation=0.5,
                                                                         hue=0.1)
                                              ], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.GaussianBlur(kernel_size=9),
                                              transforms.Normalize((0.5,), (0.5,))
                                              ])

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    ContrastiveTransformations(contrast_transforms, n_views=2)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32))
])

train_dataset.transform = train_transform
test_dataset.transform = test_transform

m = len(train_dataset)

train_val_ratio = 0.2
train_data, val_data = random_split(train_dataset, [int(m - m * train_val_ratio), int(m * train_val_ratio)])
batch_size = 256

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

visualization_batch_size = 200
test_loader = iter(torch.utils.data.DataLoader(test_dataset, batch_size=visualization_batch_size, shuffle=True))


class Encoder(nn.Module):

    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channel, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(True),
            nn.Conv2d(256, 512, 3, stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ELU(True),
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 512, 128),
            nn.ELU(True),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        x = self.cnn(x)
        # print("enc: ", x.shape)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def loss_info_nce(self, x_batch, temperature=1.):
        imgs = torch.cat(x_batch, dim=0)

        z = self.cnn(imgs)
        z = self.flatten(z)
        z = self.fc(z)

        cos_sim = F.cosine_similarity(z[:, None, :], z[None, :, :], dim=-1)
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=device)
        cos_sim.masked_fill_(self_mask, -9e16)
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        cos_sim = cos_sim / temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        return nll


lr = 0.0003

torch.manual_seed(0)

d = 2
encoder = Encoder(latent_dim=d).to(device)

optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-05)


def train_epoch(encoder, data_loader, optimizer):
    encoder.train()
    train_loss = []

    for image_batch, _ in data_loader:
        image_batch = [v.to(device) for v in image_batch]
        loss = encoder.loss_info_nce(image_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    return np.mean(train_loss)


def test_epoch(encoder, data_loader):
    encoder.eval()

    with torch.no_grad():
        losses = []

        for image_batch, _ in data_loader:
            image_batch = [v.to(device) for v in image_batch]
            loss = encoder.loss_info_nce(image_batch)

            losses.append(loss.item())

    return np.mean(losses)


# from https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points
def imscatter(x, y, image, zoom=0.5):
    ax = plt.gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    i = 0
    for x0, y0 in zip(x, y):
        im = OffsetImage(image[i], zoom=zoom)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
        i += 1
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def plot_ae_outputs(encoder, test_loader):
    plt.cla()

    try:
        img, _ = test_loader.next()
    except:
        test_loader = iter(torch.utils.data.DataLoader(test_dataset, batch_size=visualization_batch_size, shuffle=True))
        img, _ = test_loader.next()

    encoder.eval()

    with torch.no_grad():
        z_batch = encoder(img.to(device))

    img = einops.rearrange(img, "b c h w -> b h w c").cpu().numpy()
    z_batch = z_batch.cpu().numpy()

    imscatter(z_batch[:, 0], z_batch[:, 1], img)

    plt.title('context encoding images')
    plt.pause(0.0001)


num_epoch = 100
diz_loss = {"train_loss": [], "val_loss": []}

plt.Figure(figsize=(16, 5))

plt.subplot(211)
plot_ae_outputs(encoder, test_loader)
plt.pause(0.0001)

for epoch in range(num_epoch):
    train_loss = train_epoch(encoder, train_loader, optimizer)

    val_loss = test_epoch(encoder, valid_loader)

    print(f"EPOCH {epoch + 1}/{num_epoch} train loss {train_loss}, val_loss {val_loss}")

    diz_loss["train_loss"].append(train_loss)
    diz_loss["val_loss"].append(val_loss)

    plt.subplot(211)
    plot_ae_outputs(encoder, test_loader)

    plt.subplot(212)
    plt.cla()
    plt.plot(diz_loss['train_loss'], label='Train')
    plt.plot(diz_loss['val_loss'], label='Valid')
    # plt.semilogy(diz_loss['train_loss'], label='Train')
    # plt.semilogy(diz_loss['val_loss'], label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.pause(0.0001)

# Plot losses
visualization_batch_size = 2000
test_loader = iter(torch.utils.data.DataLoader(test_dataset, batch_size=visualization_batch_size, shuffle=True))

plt.figure(figsize=(16, 5))
plt.subplot(211)
plot_ae_outputs(encoder, test_loader)
plt.subplot(212)
plt.semilogy(diz_loss['train_loss'], label='Train')
plt.semilogy(diz_loss['val_loss'], label='Valid')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
# plt.grid()
plt.legend()
# plt.title('loss')

plt.show()
