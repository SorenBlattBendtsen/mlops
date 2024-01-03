if __name__ == "__main__":
    # Get the data and process it
    import torch

    path = "./data/raw/"
    train_images, train_labels = [], []
    for i in range(0, 10):
        train_images.append(torch.load(path + "train_images_" + str(i) + ".pt"))
        train_labels.append(torch.load(path + "train_target_" + str(i) + ".pt"))

    test_images = torch.load(path + "test_images.pt")
    test_labels = torch.load(path + "test_target.pt")

    # stack the tensors
    train_images = torch.cat(train_images)
    train_labels = torch.cat(train_labels)

    # normalize the images to have mean 0 and std 1
    train_images = (train_images - train_images.mean()) / train_images.std()
    test_images = (test_images - test_images.mean()) / test_images.std()

    # save tensors to procceced
    path = "./data/processed/"
    torch.save(train_images, path + "train_images.pt")
    torch.save(train_labels, path + "train_target.pt")
    torch.save(test_images, path + "test_images.pt")
    torch.save(test_labels, path + "test_target.pt")
