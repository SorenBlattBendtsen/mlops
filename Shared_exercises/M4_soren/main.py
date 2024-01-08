import click
import torch
import matplotlib.pyplot as plt
from model import MyAwesomeModel
from torch import nn, optim

from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=10, help="epochs to use for training")
def train(lr, epochs):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_set, _ = mnist()

    train_losses, test_losses = [], []
    model.train()

    for e in range(epochs):
        for images, labels in train_set:
            # flatten images
            images = images.view(images.shape[0], -1)

            # clear gradients
            optimizer.zero_grad()

            # forward pass
            output = model(images)

            # calculate loss
            loss = criterion(output, labels)

            # backward pass
            loss.backward()

            # update weights
            optimizer.step()

            # update loss
            train_losses.append(loss.item())

    # save plot of training loss
    fig_path = "/Users/sorenbendtsen/Documents/GitHub/mlops/day2_soren_cookiecutter_project/reports/figures"
    plt.plot(train_losses)
    plt.savefig(fig_path + "/train_losses.png")

    # save model
    model_path = "/Users/sorenbendtsen/Documents/GitHub/mlops/day2_soren_cookiecutter_project/day2_soren_cookiecutter_project/models/saved_models"
    torch.save(model.state_dict(), model_path + "/ffn_model_checkpoint.pt")


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)
    _, test_set = mnist()
    test_accuracy = []
    with torch.no_grad():
        model.eval()
        for images, labels in test_set:
            # flatten images
            images = images.view(images.shape[0], -1)
            # forward pass
            output = model(images)

            # probabilities
            ps = torch.exp(output)

            # calculate accuracy
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            test_accuracy.append(accuracy.item())

    # print accuracy
    print(f"Accuracy: {sum(test_accuracy)/len(test_accuracy)}")
    # save plot of test accuracy
    fig_path = "/Users/sorenbendtsen/Documents/GitHub/mlops/day2_soren_cookiecutter_project/reports/figures"
    plt.plot(test_accuracy)
    plt.savefig(fig_path + "/test_accuracy.png")


cli.add_command(
    train
)  # in terminal, run: python train_model.py train --lr 1e-4 --epochs 10
cli.add_command(
    evaluate
)  # in terminal, run: python train_model.py evaluate models/saved_models/ffn_model_checkpoint.pt


if __name__ == "__main__":
    cli()
