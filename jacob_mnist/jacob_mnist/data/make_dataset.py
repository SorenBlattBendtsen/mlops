import torch
import os


def main(path="./data/", empty_train=False, empty_test=False):
    # Get the data and process it
    path_raw = os.path.join(path, "raw/")
    train_images, train_labels = [], []
    for i in range(0, 10):
        train_images.append(torch.load(path_raw + "train_images_" + str(i) + ".pt"))
        train_labels.append(torch.load(path_raw + "train_target_" + str(i) + ".pt"))

    test_images = torch.load(path_raw + "test_images.pt")
    test_labels = torch.load(path_raw + "test_target.pt")

    if empty_train:
        train_images = []
        train_labels = []

    if not train_images or not train_labels:
        raise ValueError("Train images or labels are empty.")

    # stack the tensors
    train_images = torch.cat(train_images)
    train_labels = torch.cat(train_labels)

    # normalize the images to have mean 0 and std 1
    train_images = (train_images - train_images.mean()) / train_images.std()
    test_images = (test_images - test_images.mean()) / test_images.std()

    if empty_test:
        test_images = None
        test_labels = None

    if test_images is None or test_labels is None:
        raise ValueError("Test images or labels are empty.")

    # save tensors to procceced
    path_processed = os.path.join(path, "processed/")
    torch.save(train_images, path_processed + "train_images.pt")
    torch.save(train_labels, path_processed + "train_target.pt")
    torch.save(test_images, path_processed + "test_images.pt")
    torch.save(test_labels, path_processed + "test_target.pt")


if __name__ == "__main__":
    main()
