import pytest
import torch

from jacob_mnist.models.model import MyAwesomeModel

class TestMyAwesomeModel:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = MyAwesomeModel()

    @pytest.mark.parametrize("batch_size", [1, 32, 64, 128])
    def test_forward(self, batch_size):
        input_tensor = torch.rand((batch_size, 1, 28, 28))  # Input tensor with shape (batch_size, 1, 28, 28)
        output = self.model(input_tensor)
        assert output.shape == (
            batch_size,
            10,
        ), f"Output shape for batch size {batch_size} should be ({batch_size}, 10)"

    def test_training_step(self):
        images = torch.rand((32, 28, 28))  # Input tensor with shape (32, 1, 28, 28)
        labels = torch.randint(0, 10, (32,))  # Random labels
        batch = (images, labels)
        loss = self.model.training_step(batch)
        assert isinstance(loss, torch.Tensor)  # Assert that loss is a tensor
