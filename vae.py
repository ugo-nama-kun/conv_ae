import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("selected device: ", device)

data_dir = "dataset"

train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)

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
			nn.Conv2d(3, 128, 3, stride=2, padding=1),
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
		)

		self.fc_loc = nn.Linear(128, latent_dim)
		self.fc_scale = nn.Linear(128, latent_dim)

		self.N = torch.distributions.Normal(0, 1)
		self.N.loc = self.N.loc.cuda()
		self.N.scale = self.N.scale.cuda()

		self.kl = 0

	def forward(self, x):
		x = self.cnn(x)
		# print("enc: ", x.shape)
		x = self.flatten(x)
		x = self.fc(x)
		mu = self.fc_loc(x)
		sigma = torch.exp(self.fc_scale(x))

		z = mu + sigma * self.N.sample(mu.shape)

		self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 0.5).sum()
		return z


class Decoder(nn.Module):

	def __init__(self, latent_dim):
		super(Decoder, self).__init__()

		self.fc = nn.Sequential(
			nn.Linear(latent_dim, 128),
			nn.BatchNorm1d(128),
			nn.ELU(True),
			nn.Linear(128, 3 * 3 * 512),
			nn.BatchNorm1d(3 * 3 * 512),
			nn.ELU(True)
		)

		self.unflatten = nn.Unflatten(dim=1, unflattened_size=(512, 3, 3))

		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(512, 256, 4, stride=2, output_padding=0),
			nn.BatchNorm2d(256),
			nn.ELU(True),
			nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
			nn.BatchNorm2d(128),
			nn.ELU(True),
			nn.ConvTranspose2d(128, 3, 3, stride=2, padding=1, output_padding=1)
		)

	def forward(self, x):
		x = self.fc(x)
		x = self.unflatten(x)
		x = self.deconv(x)
		x = torch.sigmoid(x)
		return x


class VariationalAutoEncoder(nn.Module):
	def __init__(self, latent_dim):
		super(VariationalAutoEncoder, self).__init__()
		self.encoder = Encoder(latent_dim)
		self.decoder = Decoder(latent_dim)

	def forward(self, x):
		x = x.to(device)
		z = self.encoder(x)
		return self.decoder(z)

	@property
	def kl(self):
		return self.encoder.kl


lr = 0.001

torch.manual_seed(0)

beta = 0.1  # For beta-VAE
d = 256
vae = VariationalAutoEncoder(latent_dim=d).to(device)

optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-05)


def train_epoch(vae, date_loader, optimizer):
	vae.train()
	train_loss = []

	for x, _ in date_loader:
		x = x.to(device)

		x_recover = vae(x)

		# print(latent.shape, x_recover.shape, image_batch.shape)

		loss = ((x - x_recover)**2).sum() + beta * vae.kl

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		train_loss.append(loss.item())

	return np.mean(train_loss)


def test_epoch(vae, data_loader):
	vae.eval()
	val_loss = 0.0

	with torch.no_grad():
		for x, _ in data_loader:
			x = x.to(device)

			x_recover = vae(x)

			loss = ((x - x_recover)**2).sum() + beta * vae.kl
			val_loss += loss.item()

	return val_loss / len(data_loader.dataset)


def plot_ae_outputs(encoder, decoder, n=10):
	plt.clf()
	# targets = test_dataset.targets.numpy()
	# t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}
	for i in range(n):
		ax = plt.subplot(2, n, i + 1)
		img = test_dataset[random.randint(0, len(test_dataset))][0].unsqueeze(0).to(device)
		encoder.eval()
		decoder.eval()

		with torch.no_grad():
			rec_img = decoder(encoder(img))

			# img = torch.permute(img, (0, 2, 3, 1))
			# rec_img = torch.permute(rec_img, (0, 2, 3, 1))
			img = img.permute((0, 2, 3, 1))
			rec_img = rec_img.permute((0, 2, 3, 1))

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


num_epoch = 100
diz_loss = {"train_loss": [], "val_loss": []}
plt.figure(figsize=(16, 4.5))

for epoch in range(num_epoch):

	train_loss = train_epoch(vae, train_loader, optimizer)

	val_loss = test_epoch(vae, test_loader)

	print(f"EPOCH {epoch + 1}/{num_epoch} train loss {train_loss}, val_loss {val_loss}")

	diz_loss["train_loss"].append(train_loss)
	diz_loss["val_loss"].append(val_loss)

	plot_ae_outputs(vae.encoder, vae.decoder, n=10)


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
