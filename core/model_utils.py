import torch.nn as nn
from torchvision import models


class CustomCNN(nn.Module):
    """
    Petit CNN entraînable from scratch.

    Ce modèle sert de baseline pédagogique pour comparer un réseau appris
    entièrement sur le dataset de plaies avec les architectures pré-entraînées
    sur ImageNet.
    """

    def __init__(self, num_classes, dropout_rate=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 112 -> 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 56 -> 28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 28 -> 14
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def build_model(architecture, num_classes, pretrained=True, dropout_rate=0.3):
    """Construit un CNN pré-entraîné avec une tête adaptée au nombre de classes."""
    architecture = architecture.lower()

    if architecture == "custom_cnn":
        if pretrained:
            print("Attention : custom_cnn ne supporte pas de poids pré-entraînés. Le modèle sera entraîné from scratch.")

        return CustomCNN(
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )

    if architecture == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16(weights=weights)

        # Remplace la dernière couche ImageNet par la classification des plaies.
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

        return model

    if architecture == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)

        # ResNet utilise l'attribut fc pour sa tête de classification.
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

        return model

    if architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        # EfficientNet stocke sa tête dans model.classifier.
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

        return model

    if architecture == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)

        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

        return model

    raise ValueError(f"Architecture non supportée : {architecture}")


def freeze_backbone(model, architecture):
    """Gèle le backbone et garde seulement la tête entraînable."""
    architecture = architecture.lower()

    if architecture.startswith("resnet"):
        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            param.requires_grad = True

    elif architecture == "efficientnet_b0":
        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True

    elif architecture == "vgg16":
        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier[6].parameters():
            param.requires_grad = True

    elif architecture == "mobilenet_v3_large":
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif architecture == "custom_cnn":
        raise ValueError(
            "custom_cnn est entraîné from scratch : il ne supporte pas freeze_backbone.")

    else:
        raise ValueError(
            f"Architecture non supportée pour freeze_backbone : {architecture}")

    return model


def unfreeze_last_blocks(model, architecture):
    """Dégèle les derniers blocs pour un fine-tuning léger."""
    architecture = architecture.lower()

    if architecture == "resnet50":
        for param in model.layer4.parameters():
            param.requires_grad = True

    elif architecture == "efficientnet_b0":
        for param in model.features[-2:].parameters():
            param.requires_grad = True

    elif architecture == "vgg16":
        for param in model.features[-8:].parameters():
            param.requires_grad = True

    elif architecture == "mobilenet_v3_large":
        for param in model.features[-3:].parameters():
            param.requires_grad = True

    elif architecture == "custom_cnn":
        raise ValueError(
            "custom_cnn est entraîné from scratch : il ne supporte pas fine_tune.")

    else:
        raise ValueError(
            f"Architecture non supportée pour unfreeze_last_blocks : {architecture}")

    return model


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())
