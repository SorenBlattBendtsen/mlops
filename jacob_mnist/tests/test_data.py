import torch
import os
import jacob_mnist.data.make_dataset as make_dataset
from tests import _PATH_DATA
import pytest

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data directory does not exist. Due to DVC")
class TestMakeDataset:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.raw_path = os.path.join(_PATH_DATA, "raw/")
        self.processed_path = os.path.join(_PATH_DATA, "processed/")
        yield
        # os.remove(self.processed_path + "train_images.pt")
        # os.remove(self.processed_path + "train_target.pt")
        # os.remove(self.processed_path + "test_images.pt")
        # os.remove(self.processed_path + "test_target.pt")

    def test_train_images(self):
        make_dataset.main(_PATH_DATA)
        train_images = torch.load(self.processed_path + "train_images.pt")
        assert train_images.mean() == pytest.approx(0, abs = 1e-3), "Mean of train_images should be 0"
        assert train_images.std() == pytest.approx(1), "Standard deviation of train_images should be 1"
        assert len(train_images) == 50000, "Length of train_images should be 50000"
        assert train_images[0].shape == (28, 28), "Shape of the first image in train_images should be (1, 28, 28)"

    def test_train_labels(self):
        make_dataset.main(_PATH_DATA)
        train_labels = torch.load(self.processed_path + "train_target.pt")
        assert isinstance(train_labels, torch.Tensor), "train_labels should be an instance of torch.Tensor"
        assert len(train_labels) == 50000, "Length of train_labels should be 50000"
        assert set(train_labels.tolist()) == set(range(10)), "train_labels should contain all numbers from 0 to 9"

    def test_test_images(self):
        make_dataset.main(_PATH_DATA)
        test_images = torch.load(self.processed_path + "test_images.pt")
        assert test_images.mean() == pytest.approx(0, abs = 1e-3), "Mean of test_images should be 0"
        assert test_images.std() == pytest.approx(1), "Standard deviation of test_images should be 1"
        assert len(test_images) == 5000, "Length of test_images should be 5000"
        assert test_images[0].shape == (28, 28), "Shape of the first image in test_images should be (1, 28, 28)"

    def test_test_labels(self):
        make_dataset.main(_PATH_DATA)
        test_labels = torch.load(self.processed_path + "test_target.pt")
        assert isinstance(test_labels, torch.Tensor), "test_labels should be an instance of torch.Tensor"
        assert len(test_labels) == 5000, "Length of test_labels should be 5000"
        assert set(test_labels.tolist()) == set(range(10)), "test_labels should contain all numbers from 0 to 9"

    def test_empty_train(self):
        with pytest.raises(ValueError, match="Train images or labels are empty."):
            make_dataset.main(_PATH_DATA, empty_train=True)
    
    def test_empty_test(self):
        with pytest.raises(ValueError, match="Test images or labels are empty."):
            make_dataset.main(_PATH_DATA, empty_test=True)