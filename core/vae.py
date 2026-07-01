import torch
import torch.nn as nn


class ConvVAE(nn.Module):
    """
    Convolutional variational autoencoder for wound image OOD detection.

    The model learns a probabilistic latent space with mean and log-variance.
    This makes it possible to combine reconstruction error with latent
    uncertainty through the KL divergence term.
    """

    def __init__(self, latent_dim=256):
        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)

        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 7 * 7),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        features = self.encoder(x)
        features = self.flatten(features)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decoder(z)
        x = x.view(-1, 512, 7, 7)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


def reconstruction_error(images, reconstructions, reduction="mean"):
    errors = torch.mean((images - reconstructions) ** 2, dim=(1, 2, 3))

    if reduction == "none":
        return errors

    if reduction == "mean":
        return errors.mean()

    raise ValueError(f"Unsupported reduction: {reduction}")


def kl_divergence(mu, logvar, reduction="mean"):
    kl = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim=1,
    )

    if reduction == "none":
        return kl

    if reduction == "mean":
        return kl.mean()

    raise ValueError(f"Unsupported reduction: {reduction}")


def vae_loss(images, reconstructions, mu, logvar, beta=0.001):
    recon = reconstruction_error(
        images,
        reconstructions,
        reduction="mean",
    )
    kl = kl_divergence(mu, logvar, reduction="mean")
    total = recon + beta * kl

    return total, recon, kl


def combined_ood_score(reconstruction_errors, kl_values, alpha=0.001):
    return reconstruction_errors + alpha * kl_values
