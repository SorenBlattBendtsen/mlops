import torch
import argparse
from models.model import MyAwesomeModel


# add arguments to parser
parser = argparse.ArgumentParser()
parser.add_argument("model_checkpoint")
parser.add_argument("data_path")


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([torch.exp(model(batch.view(batch.shape[0], -1))) for batch in dataloader], 0)


# if main
if __name__ == "__main__":
    # Get the data and process it

    # get arguments
    args = parser.parse_args()
    model_checkpoint = args.model_checkpoint
    data_path = args.data_path
    model = MyAwesomeModel()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)

    # load data
    test_images = torch.load(data_path + "test_images.pt")

    # create dataloaders
    test = torch.utils.data.DataLoader(test_images, batch_size=32, shuffle=True)
    predictions = predict(model, test)
    print(predictions)
