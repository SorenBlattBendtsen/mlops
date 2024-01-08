import torch
from jacob_mnist.models.model import MyAwesomeModel
import pytest


class TestMyAwesomeModel:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = MyAwesomeModel()

    def test_forward(self):
        input_tensor = torch.rand((1, 1, 28, 28))  # Input tensor with shape (1, 1, 28, 28)
        output = self.model(input_tensor)
        assert output.shape == (1, 10), "Output shape should be (1, 10)"
