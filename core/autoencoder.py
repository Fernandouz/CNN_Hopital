import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Autoencoder convolutif pour la reconstruction d'images de plaies.

    Le modèle apprend une représentation latente compressée des images du domaine connu.
    L'erreur de reconstruction pourra ensuite être utilisée comme score d'anomalie.
    """

    def __init__(self, latent_dim=256):
        super().__init__()

        self.latent_dim = latent_dim

        # Entrée : 3 x 224 x 224
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2,
                      padding=1),   # 32 x 112 x 112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2,
                      padding=1),  # 64 x 56 x 56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2,
                      padding=1),  # 128 x 28 x 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2,
                      padding=1),  # 256 x 14 x 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2,
                      padding=1),  # 512 x 7 x 7
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten()

        self.fc_encoder = nn.Sequential(
            nn.Linear(512 * 7 * 7, latent_dim),
            nn.ReLU(inplace=True),
        )

        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 7 * 7),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4,
                               stride=2, padding=1),  # 256 x 14 x 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4,
                               stride=2, padding=1),  # 128 x 28 x 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4,
                               stride=2, padding=1),  # 64 x 56 x 56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2,
                               padding=1),   # 32 x 112 x 112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2,
                               padding=1),    # 3 x 224 x 224
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        z = self.fc_encoder(x)
        return z

    def decode(self, z):
        x = self.fc_decoder(z)
        x = x.view(-1, 512, 7, 7)
        x = self.decoder(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction


def reconstruction_error(images, reconstructions, reduction="mean"):
    """
    Calcule l'erreur de reconstruction MSE par image.

    Retour :
    - reduction="none" : tenseur de taille [batch_size]
    - reduction="mean" : moyenne du batch
    """
    errors = torch.mean((images - reconstructions) ** 2, dim=(1, 2, 3))

    if reduction == "none":
        return errors

    if reduction == "mean":
        return errors.mean()

    raise ValueError(f"Réduction non supportée : {reduction}")
