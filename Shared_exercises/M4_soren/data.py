import torch


def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    path = "/Users/sorenbendtsen/Documents/GitHub/dtu_mlops/data/corruptmnist/"
    train_images, train_labels = [], []
    for i in range(1, 6):
        train_images.append(torch.load(path + "train_images_" + str(i) + ".pt"))
        train_labels.append(torch.load(path + "train_target_" + str(i) + ".pt"))

    test_images = torch.load(path + "test_images.pt")
    test_labels = torch.load(path + "test_target.pt")

    # stack the tensors
    train_images = torch.cat(train_images)
    train_labels = torch.cat(train_labels)

    # convert to torch tensors
    train = torch.utils.data.TensorDataset(train_images, train_labels)
    test = torch.utils.data.TensorDataset(test_images, test_labels)

    # create dataloaders
    train = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    test = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)

    return train, test
