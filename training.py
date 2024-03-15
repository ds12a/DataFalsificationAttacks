from util import LossInfo
import torch
import torch.optim as optim

def trainSingle(
    model, optimizer, criterion, num_epochs, training_loader, testing_loader
):
    for epoch in range(num_epochs):
        train_loss = LossInfo()

        for training in training_loader:
            model.zero_grad()

            inputs, target = training
            outputs = model(inputs)

            loss = criterion(outputs, target)

            loss.backward()

            optimizer.step()

            train_loss.add(loss.item())

        if epoch % 10 == 0:
            test_loss = LossInfo()

            with torch.no_grad():
                for test in testing_loader:
                    inputs, target = test

                    outputs = model(inputs)

                    loss = criterion(outputs, target)

                    test_loss.add(loss.item())

                # print(inputs, "\n --> \n", outputs)

            print(f"Validation: {test_loss.getInfo()}")

        print(f"Epoch [{epoch+1}/{num_epochs}] | {train_loss.getInfo()}")


def trainGAN(
    modelG,
    modelD,
    optimizerG,
    optimizerD,
    criterion,
    num_epochs,
    training_loader,
    window,
    latent_space,
    batch_size,
):
    for epoch in range(num_epochs):
        for i, training in enumerate(training_loader):
            # Train Discriminator
            modelD.zero_grad()

            # Use existing data
            inputs, target = training
            outputs, _ = modelD(inputs)

            lossD = criterion(outputs, target)

            lossD.backward()

            detectorDataLoss = lossD.mean().item()

            # Generate fake data
            noise = torch.randn(batch_size, latent_space)
            fake = modelG(noise).detach()

            outputs, _ = modelD(fake)
            target = torch.ones(batch_size)

            lossD = criterion(outputs, target)
            lossD.backward()
            detectorGeneratedLoss = lossD.mean().item()

            optimizerD.step()

            detectorLoss = detectorDataLoss + detectorGeneratedLoss

            # Train Generator
            modelG.zero_grad()
            target = torch.zeros(batch_size)  # Want D to think genuine

            outputs, _ = modelD(fake)

            lossG = criterion(outputs, target)
            lossG.backward()

            generatorLoss = lossG.mean().item()

            optimizerG.step()

            if i % 100 == 0:
                print(
                    f"[{epoch+1}/{num_epochs}][{i}/{len(training_loader)}] Detector Loss: {detectorLoss} | Generator Loss: {generatorLoss} "
                )


def trainGAN_AE(
    modelG,
    modelD,
    modelD_noise,
    modelE,
    optimizerG,
    optimizerD,
    optimizerD_noise,
    optimizerE,
    criterion,
    num_epochs,
    training_loader,
    window,
    latent_space,
    batch_size,
):
    for epoch in range(num_epochs):
        for i, training in enumerate(training_loader):
            # Train Discriminator
            modelD.zero_grad()

            # Use existing data
            inputs, target = training
            outputs, _ = modelD(inputs)

            lossD = criterion(outputs, target)

            lossD.backward()

            detectorDataLoss = lossD.mean().item()

            # Generate fake data
            noise = torch.randn(batch_size, latent_space)
            fake = modelG(noise).detach()

            outputs, _ = modelD(fake)
            target = torch.ones(batch_size)

            lossD = criterion(outputs, target)
            lossD.backward()
            detectorGeneratedLoss = lossD.mean().item()

            optimizerD.step()

            detectorLoss = detectorDataLoss + detectorGeneratedLoss

            # Train Generator
            modelG.zero_grad()
            target = torch.zeros(batch_size)  # Want D to think genuine

            outputs, _ = modelD(fake)

            lossG = criterion(outputs, target)
            lossG.backward()

            generatorLoss = lossG.mean().item()

            optimizerG.step()

            # Train Encoder
            modelE.zero_grad()

            inputs, target = training
            output = modelE(inputs)

            if i % 100 == 0:
                print(
                    f"[{epoch+1}/{num_epochs}][{i}/{len(training_loader)}] Detector Loss: {detectorLoss} | Generator Loss: {generatorLoss} "
                )


def trainTadGAN(
    encoder,
    decoder,
    critic_x,
    critic_z,
    criterion,
    num_epochs,
    training_loader,
    window,
    latent_space,
    batch_size,
    lr=1e-6,
    n_critics=5
):
    optimE = optim.Adam(encoder.parameters(), lr=lr, betas=(0.5, 0.999))
    optimD = optim.Adam(decoder.parameters(), lr=lr, betas=(0.5, 0.999))
    optimCX = optim.Adam(critic_x.parameters(), lr=lr, betas=(0.5, 0.999))
    optimCZ = optim.Adam(critic_z.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for x in training_loader:
            # Train critic x (real vs. generated)
            optimCX.zero_grad()
            
            outputs = critic_x(x)

            
