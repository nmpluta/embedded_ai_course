import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
from typing import Tuple
import tqdm
import matplotlib.pyplot as plt
import platform
import time
import numpy as np


class TimeMeasurement:
    def __init__(self, context_name: str, frames: int) -> None:
        self.context_name: str = context_name
        self.frames: int = frames
        self.begin: float = None
        self.end: float = None

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()

    @property
    def time(self) -> float:
        if self.begin is None or self.end is None:
            raise RuntimeError()
        return self.end - self.begin

    @property
    def fps(self):
        return self.frames / self.time

    def __str__(self) -> str:
        t = self.time
        h = t // 60
        min = (t - h*60) // 60
        s = int(t - h*60 - min*60)
        ms = int((t - np.floor(t))*1000)

        return f"Execution time: {h}:{min}:{s}:{ms}, processed {self.frames} frames, throughput: {self.fps} fps."

    def __repr__(self) -> str:
        t = self.time
        h = t // 60
        min = (t - h*60) // 60
        s = int(t - h*60 - min*60)
        ms = int((t - np.floor(t))*1000)

        return f'TimeMeasurement(context="{self.context_name}","{h}:{min}:{s}:{ms}", frames={self.frames}, throughput={self.fps})'


def display_tensor_as_img(t: torch.Tensor, title=''):
    t = t.reshape((1,) + t.shape[-2:])

    for i in range(t.shape[0]):
        plt.imshow(t[i,:,:])
        plt.title(title + str(i))
        plt.show()


class BaseMetic(ABC):

    @abstractmethod
    def __call__(self, y_pred, y_ref) -> Any:
        raise NotImplementedError()


class AccuracyMetric(BaseMetic):

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, y_pred: torch.Tensor, y_ref: torch.Tensor) -> torch.Tensor:
        """
        :param y_pred: tensor of shape (batch_size, num_of_classes) type float
        :param y_ref: tensor with shape (batch_size,) and type Long
        :return: scalar tensor with accuracy metric for batch
        """
        y_pred = y_pred.argmax(1)
        cmp = y_pred == y_ref
        # scalar value
        score: torch.Tensor = cmp.sum() / cmp.shape[0]

        return score


def count_params(model: torch.nn.Module):
    num_of_params = 0
    for p in model.parameters():
        num_of_params += p.view(-1,1).shape[0]

    return num_of_params


def train_test_pass(model: torch.nn.Module,
                    data_generator,
                    criterion,
                    metric: BaseMetic,
                    optimizer: torch.optim.Optimizer = None,
                    update_period: int = None,
                    mode: str = 'test',
                    device = torch.device('cpu')) -> Tuple[torch.nn.Module, float, float]:
    """
    Train or test pass generator data through the model.

    :param model: network
    :param data_generator: data loader
    :param criterion: criterion / loss two arg function
    :param metric: metric object - two arg function
    :param optimizer: optimizer object. For test mode use None. Defaults to None
    :param update_period: number of batches of processing to update parameters. For test mode use None. Defaults to None
    :param mode: one of ['train', 'test']. Test mode is for evaluation. Train for training (includes gradient propagation), defaults to 'test'
    :param device: device to execute on.
    :return: model, loss_value, metric_value
    """
    print(f"Running on platform: {platform.platform()}, "
          f"machine: {platform.machine()}, "
          f"python_version: {platform.python_version()}, "
          f"processor: {platform.processor()}, "
          f"system: {platform.system()}, "
          )

    # change model mode to train or test
    if mode == 'train':
        model.train(True)

    elif mode == 'test':
        model.eval()

    else:
        raise RuntimeError("Unsupported mode.")

    # move model to device
    model = model.to(device)

    # reset model parameters' gradients with optimizer
    if mode == 'train':
        optimizer.zero_grad()

    total_loss: float = 0.0
    total_accuracy: float = 0.0
    samples_num: int = 0

    for i, (X, y_ref) in tqdm.tqdm(enumerate(data_generator),):
        # convert tensors to device
        X = X.to(device)
        y_ref = y_ref.to(device)

        if mode == 'train':
            # process by network
            y_pred = model(X)
        else:
            with torch.no_grad():
                y_pred = model(X)

        # calculate loss
        loss: torch.Tensor = criterion(y_pred, y_ref)

        if mode == 'train':
            # designate gradient based on loss
            loss.backward()

        if mode == 'train' and (i+1) % update_period == 0:
            # update parameters with optimizer
            optimizer.step()
            # gradient designation sums it's values from previous passes
            # there is needed zeroing stored values of gradient
            optimizer.zero_grad()

        # calculate accuracy
        accuracy = metric(y_pred, y_ref)

        total_loss += loss.item() * y_pred.shape[0]
        total_accuracy += accuracy.item() * y_pred.shape[0]
        samples_num += y_pred.shape[0]

    if samples_num == 0:
        return model, 0.0, 0.0

    return model, total_loss / samples_num, total_accuracy / samples_num


def training(model,
             train_loader,
             test_loader,
             loss_fcn,
             metric,
             optimizer,
             update_period,
             epoch_max,
             device) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """
    _summary_
    :param model: network
    :param data_generator: data loader
    :return: model, (loss_value, metric_value)

    :param model: network model
    :param train_loader: data loader for training
    :param test_loader: data loader for validation
    :param loss_fcn: criterion / loss two arg function
    :param metric: metric object - two arg function
    :param optimizer: optimizer object
    :param update_period: number of batches of processing to update parameters
    :param epoch_max: number of training epochs
    :param device: device to execute on.
    :return: model, dictionary with keys 'loss_train', 'loss_test', 'acc_train', 'acc_test'
    - each entry contains list of loss / metric value for given epoch
    """
    loss_train = []
    loss_test = []
    acc_train = []
    acc_test = []

    for e in range(epoch_max):
        epoch = e+1
        print(f'Epoch {epoch} / {epoch_max}: STARTED')
        print('TRAINING')
        net, loss, acc = train_test_pass(model,
                                         train_loader,
                                         loss_fcn,
                                         metric,
                                         optimizer,
                                         update_period=update_period,
                                         mode='train',
                                         device=device)
        loss_train.append(loss)
        acc_train.append(acc)

        print('VALIDATION')
        net, loss, acc = train_test_pass(model,
                                         test_loader,
                                         loss_fcn,
                                         metric,
                                         optimizer,
                                         update_period=update_period,
                                         mode='test',
                                         device=device)
        loss_test.append(loss)
        acc_test.append(acc)

        print(
            f'\rAfter epoch {epoch}: loss={loss_train[-1]:.4f} acc={acc_train[-1]:.4f} val_loss={loss_test[-1]:.4f} val_acc={acc_test[-1]:.4f}')
        print(f'Epoch {epoch} / {epoch_max}: FINISHED\n')

    return model, {'loss_train': loss_train,
                   'acc_train': acc_train,
                   'loss_test': loss_test,
                   'acc_test': acc_test}

def plot_history(history):
    plt.plot(history['loss_train'], label='train')
    plt.plot(history['loss_test'], label='test')
    plt.legend()
    plt.title("History of loss")
    plt.show()

    plt.plot(history['acc_train'], label='train')
    plt.plot(history['acc_test'], label='test')
    plt.legend()
    plt.title("History of accuracy")
    plt.show()


class ResidualBlock(nn.Module):
    def __init__(self,
                 input_channels: int,
                 intermediate_channels: int,
                 kernel_size: Union[int, Tuple[int,int]],
                 ) -> None:
        super().__init__()
        self.L1 = nn.Sequential(nn.Conv2d(in_channels=input_channels,
                                          out_channels=intermediate_channels,
                                          kernel_size=kernel_size,
                                          bias=False,
                                          padding=kernel_size//2),
                                nn.BatchNorm2d(intermediate_channels),
                                nn.ReLU()
                                )
        self.L2 = nn.Sequential(nn.Conv2d(in_channels=intermediate_channels,
                                          out_channels=input_channels,
                                          kernel_size=kernel_size,
                                          bias=False,
                                          padding=kernel_size//2),
                                nn.BatchNorm2d(input_channels),
                                nn.ReLU()
                                )

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        return torch.add(x, self.L2(self.L1(x)))

def get_train_dataset(batch_size, dataset_dir='data') -> DataLoader:
    train_dataset = datasets.MNIST(dataset_dir,
                                   train=True,
                                   download=True,
                                   transform=ToTensor())
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    return train_loader


def get_test_dataset(batch_size, dataset_dir='data') -> DataLoader:
    test_dataset = datasets.MNIST(dataset_dir,
                                  train=False,
                                  download=True,
                                  transform=ToTensor())

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    return test_loader