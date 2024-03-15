import torch.nn as nn
import torch
import matplotlib.pyplot as plt


def plotAELoss(model, normal_loader, malicious_loader, threshold):
    normalLoss, malLoss = [], []
    mse = nn.MSELoss()

    normal_correct = 0
    mal_correct = 0

    for t in normal_loader:
        inputs = t
        outputs = model(inputs)
        loss = mse(outputs, inputs)
        normalLoss.append(loss.item())
        if loss.item() < threshold:
            normal_correct += 1

    for t in malicious_loader:
        inputs = t
        outputs = model(inputs)
        loss = mse(outputs, inputs)
        malLoss.append(loss.item())
        if loss.item() >= threshold:
            mal_correct += 1

    print(
        f"Normal accuracy: {normal_correct / len(normal_loader)}, Malicious accuracy: {mal_correct / len(malicious_loader)}"
    )
    plt.figure()
    plt.hist(
        [normalLoss, malLoss],
        label=["Normal", "Malicious"],
        bins=200,
        range=(0, 400000),
    )
    plt.legend()
    plt.show()


def plotClassifierOutput(model, normal_loader, malicious_loader, threshold):
    normalOut, malOut = [], []

    normal_correct = 0
    mal_correct = 0

    for t in normal_loader:
        inputs = t
        outputs = model(inputs)
        normalOut.append(outputs.item())
        if outputs < threshold:
            normal_correct += 1

    for t in malicious_loader:
        inputs = t
        outputs = model(inputs)
        malOut.append(outputs.item())
        if outputs >= threshold:
            mal_correct += 1

    print(
        f"Normal accuracy: {normal_correct / len(normal_loader)}, Malicious accuracy: {mal_correct / len(malicious_loader)}"
    )
    plt.figure()
    plt.hist(
        [normalOut, malOut],
        label=["Normal", "Malicious"],
        bins=200,
    )
    plt.legend()
    plt.show()


class LossInfo:
    def __init__(self) -> None:
        self.sum = 0
        self.count = 0

    def add(self, loss):
        self.sum += loss
        self.count += 1

    def getInfo(self):
        return self.sum / self.count


def anomaly_score(x, modelG, modelD, batch_size, latent_space, pct=0.7):
    z = torch.randn(batch_size, latent_space)
    z_optimizer = torch.optim.Adam([z], lr=1e-2)

    loss = None

    for i in range(40):
        generated = modelG(z)
        _, featureX = modelD(x)
        _, featureZ = modelD(x)

        residual_loss = torch.sum(torch.abs(x - generated))  # Residual Loss
        discriminator_loss = torch.sum(torch.abs(featureX - featureZ))

        loss = (1 - pct) * residual_loss + pct * discriminator_loss

        loss.backward()
        z_optimizer.step()

    return loss


def plotAnomalyScores(modelG, modelD, normal_loader, malicious_loader, latent_space, threshold):
    normalOut, malOut = [], []

    normal_correct = 0
    mal_correct = 0

    for i, t in enumerate(normal_loader):
        print(i)
        inputs = t
        outputs = anomaly_score(
            inputs,
            modelG,
            modelD,
            1,
            latent_space,
        )
        normalOut.append(outputs)
        if outputs < threshold:
            normal_correct += 1

    for i, t in enumerate(malicious_loader):
        print(i)
        inputs = t
        outputs = anomaly_score(
            inputs,
            modelG,
            modelD,
            1,
            latent_space,
        )
        malOut.append(outputs)
        if outputs >= threshold:
            mal_correct += 1

    print(
        f"Normal accuracy: {normal_correct / len(normal_loader)}, Malicious accuracy: {mal_correct / len(malicious_loader)}"
    )
    with torch.no_grad():
        plt.figure()
        plt.hist(
            [normalOut, malOut],
            label=["Normal", "Malicious"],
            bins=200,
        )
        plt.legend()
        plt.show()
