import click
import torch
from model import MyAwesomeModel
import matplotlib.pyplot as plt
from data import mnist

path = "./Shared_exercises/M4/"


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=10, help="learning rate to use for training")
def train(lr, epochs):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist(path=path + "data/")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.NLLLoss()
    train_losses = []
    model.train()
    for e in range(epochs):
        for images, labels in train_set:
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

    # save plot of training loss
    plt.plot(train_losses)
    plt.savefig(path + "train_losses.png")
    # save model
    torch.save(model.state_dict(), path + "checkpoint.pth")


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
    accuracies = []
    with torch.no_grad():
        model.eval()
        for images, labels in test_set:
            images = images.view(images.shape[0], -1)
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            accuracies.append(accuracy.item())
    print(f"Accuracy: {sum(accuracies)/len(accuracies)}")


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
