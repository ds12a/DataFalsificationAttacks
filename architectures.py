import torch.nn as nn


# Relu bad
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, input_dim // 2),
        #     nn.Sigmoid(),
        #     nn.Linear(input_dim // 2, input_dim // 4),
        #     nn.Sigmoid(),
        #     nn.Linear(input_dim // 4, input_dim // 8),
        #     nn.Sigmoid(),
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(input_dim // 8, input_dim // 4),
        #     nn.Sigmoid(),
        #     nn.Linear(input_dim // 4, input_dim // 2),
        #     nn.Sigmoid(),
        #     nn.Linear(input_dim // 2, input_dim),
        #     nn.Sigmoid(),
        # )

        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, input_dim // 2),
        #     nn.LeakyReLU(),
        #     nn.Linear(input_dim // 2, input_dim // 4),
        #     nn.LeakyReLU(),
        #     nn.Linear(input_dim // 4, input_dim // 8),
        #     nn.LeakyReLU(),
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(input_dim // 8, input_dim // 4),
        #     nn.LeakyReLU(),
        #     nn.Linear(input_dim // 4, input_dim // 2),
        #     nn.LeakyReLU(),
        #     nn.Linear(input_dim // 2, input_dim),
        #     nn.Sigmoid(),
        # )

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 11),
            nn.ReLU(),
            nn.Conv1d(16, 64, 7),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((input_dim - 10 - 6) * 64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.ReLU(),
        )

        self.dLinear = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, (input_dim - 10 - 6) * 64),
            nn.ReLU(),
            nn.Unflatten(1, (64, input_dim - 10 - 6)),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 16, 7),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 11),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.dLinear(self.encoder(x.unsqueeze(1)))).squeeze(1)


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, inputs, targets, malicious):
        loss = (inputs - targets) ** 2
        return (
            (malicious) * (1 - loss.mean(dim=1)) + (1 - malicious) * (loss.mean(dim=1))
        ).mean() * -1  # Maybe log?


class FCClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FCClassifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class CNNlassifier(nn.Module):
    def __init__(self, feature_factor=4):
        super(CNNlassifier, self).__init__()

        # self.model = nn.Sequential(
        #     nn.Conv1d(1, feature_factor * 2, 11),
        #     nn.LeakyReLU(),
        #     nn.Conv1d(feature_factor * 2, feature_factor * 4, 7),
        #     nn.BatchNorm1d(feature_factor * 4),
        #     nn.LeakyReLU(),
        #     nn.MaxPool1d(3, stride=1),
        #     nn.Conv1d(feature_factor * 4, feature_factor * 4, 5),
        #     nn.BatchNorm1d(feature_factor * 4),
        #     nn.LeakyReLU(),
        #     nn.MaxPool1d(3, stride=2),
        #     nn.Flatten(),
        #     nn.Linear(512, 64),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(64, 16),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(16, 4),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(4, 1),
        #     nn.Sigmoid(),
        # )

        self.conv = nn.Sequential(
            nn.Conv1d(1, feature_factor * 2, 11),
            nn.LeakyReLU(),
            nn.Conv1d(feature_factor * 2, feature_factor * 4, 7),
            nn.BatchNorm1d(feature_factor * 4),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        self.predictor = nn.Sequential(
            nn.Linear(384, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.conv(x.unsqueeze(1))
        return (self.predictor(features).squeeze(1), features)


class Generator(nn.Module):
    def __init__(self, output_dim, feature_factor=16):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose1d(1, feature_factor * 8, 4, 1, 0, bias=False),
            nn.BatchNorm1d(feature_factor * 8),
            nn.ReLU(True),
            nn.ConvTranspose1d(
                feature_factor * 8, feature_factor * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm1d(feature_factor * 4),
            nn.ReLU(True),
            nn.ConvTranspose1d(
                feature_factor * 4, feature_factor * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm1d(feature_factor * 2),
            nn.ReLU(True),
            nn.ConvTranspose1d(feature_factor * 2, feature_factor, 4, 2, 1, bias=False),
            nn.BatchNorm1d(feature_factor),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(1664, 128),
            nn.ReLU(True),
            nn.Linear(128, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x.unsqueeze(1)).squeeze(1)

    
