import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn

import torch.nn.functional as F
import torch.optim as optim

data_dir = "dataset"

# train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
# test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)

train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)

# print(test_dataset.targets)

train_transform = transforms.Compose([
	transforms.ToTensor(),
])

test_transform = transforms.Compose([
	transforms.ToTensor(),
])

train_dataset.transform = train_transform
test_dataset.transform = test_transform

m=len(train_dataset)

train_val_ratio = 0.2
train_data, val_data = random_split(train_dataset, [int(m - m*train_val_ratio), int(m*train_val_ratio)])
batch_size = 256

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class Encoder(nn.Module):

	def __init__(self, latent_dim):
		super(Encoder, self).__init__()

		self.cnn = nn.Sequential(
			nn.Conv2d(3, 8, 3, stride=2, padding=1),
			nn.ELU(True),
			nn.Conv2d(8, 16, 3, stride=2, padding=1),
			nn.ELU(True),
			nn.Conv2d(16, 32, 3, stride=2, padding=0),
			nn.ELU(True),
		)

		self.flatten = nn.Flatten(start_dim=1)

		self.fc = nn.Sequential(
			nn.Linear(3 * 3 * 32, 128),
			nn.ELU(True),
			nn.Linear(128, latent_dim)
		)

	def forward(self, x):
		x = self.cnn(x)
		# print("enc: ", x.shape)
		x = self.flatten(x)
		x = self.fc(x)
		return x


class Decoder(nn.Module):

	def __init__(self, latent_dim):
		super(Decoder, self).__init__()

		self.fc = nn.Sequential(
			nn.Linear(latent_dim, 128),
			nn.ReLU(True),
			nn.Linear(128, 3 * 3 * 32),
			nn.ReLU(True)
		)

		self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1),
			nn.ReLU(True),
			nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
			nn.ReLU(True),
			nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1)
		)

	def forward(self, x):
		x = self.fc(x)
		x = self.unflatten(x)
		x = self.deconv(x)
		x = torch.sigmoid(x)
		return x


loss_fn = torch.nn.MSELoss()

lr = 0.0003

torch.manual_seed(0)

d = 10
encoder = Encoder(latent_dim=d)
decoder = Decoder(latent_dim=d)
params_to_optimize = [
	{"params": encoder.parameters()},
	{"params": decoder.parameters()}
]

optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)


def train_epoch(encoder, decoder, date_loader, loss_fn, optimizer):
	encoder.train()
	decoder.train()
	train_loss = []

	for image_batch, _ in date_loader:
		latent = encoder(image_batch)
		x_recover = decoder(latent)

		# print(latent.shape, x_recover.shape, image_batch.shape)

		loss = loss_fn(x_recover, image_batch)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		train_loss.append(loss.item())

	return np.mean(train_loss)


def test_epoch(encoder, decoder, data_loader, loss_fn):
	encoder.eval()
	decoder.eval()

	with torch.no_grad():
		conc_out = []
		conc_label = []

		for image_batch, _ in data_loader:
			latent = encoder(image_batch)
			x_recover = decoder(latent)

			conc_out.append(x_recover)
			conc_label.append(image_batch)

		conc_out = torch.cat(conc_out)
		conc_label = torch.cat(conc_label)

		val_loss = loss_fn(conc_out, conc_label)

	return val_loss.data


def plot_ae_outputs(encoder, decoder, n=10):
	plt.clf()
	# targets = test_dataset.targets.numpy()
	# t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}
	targets = test_dataset.targets
	t_idx = {i: targets[i] for i in range(n)}
	for i in range(n):
		ax = plt.subplot(2, n, i + 1)
		img = test_dataset[t_idx[i]][0].unsqueeze(0)
		encoder.eval()
		decoder.eval()

		with torch.no_grad():
			rec_img = decoder(encoder(img))

			img = torch.permute(img, (0, 2, 3, 1))

			rec_img = torch.permute(rec_img, (0, 2, 3, 1))

		plt.imshow(img.cpu().squeeze().numpy())
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		if i == n // 2:
			ax.set_title('Original images')
		ax = plt.subplot(2, n, i + 1 + n)
		plt.imshow(rec_img.cpu().squeeze().numpy())
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		if i == n // 2:
			ax.set_title('Reconstructed images')
	plt.pause(0.0001)


num_epoch = 30
diz_loss = {"train_loss": [], "val_loss": []}
plt.figure(figsize=(16, 4.5))

for epoch in range(num_epoch):

	train_loss = train_epoch(encoder, decoder, train_loader, loss_fn, optimizer)

	val_loss = test_epoch(encoder, decoder, test_loader, loss_fn)

	print(f"EPOCH {epoch + 1}/{num_epoch} train loss {train_loss}, val_loss {val_loss}")

	diz_loss["train_loss"].append(train_loss)
	diz_loss["val_loss"].append(val_loss)

	plot_ae_outputs(encoder, decoder, n=10)


# Plot losses
plt.figure(figsize=(10, 8))
plt.semilogy(diz_loss['train_loss'], label='Train')
plt.semilogy(diz_loss['val_loss'], label='Valid')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
#plt.grid()
plt.legend()
#plt.title('loss')
plt.show()
