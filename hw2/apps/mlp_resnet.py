import sys
try:
    sys.path.append("../python")
except:
    pass
import needle as ndl
from needle import nn, data
import numpy as np
import time
import os

np.random.seed(0)


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    # BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim)
            )
        ),
        nn.ReLU()
    )

    # END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    # BEGIN YOUR SOLUTION

    block_list = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    for _ in range(num_blocks):
        block_list.append(ResidualBlock(hidden_dim, hidden_dim//2,
                                        norm, drop_prob))
    block_list.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*block_list)
    # END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
        err_cum = 0
        loss_cum = 0
        num_sample = 0
        for batch in dataloader:
            batch_X, batch_y = batch
            loss, err = loss_err(model(nn.Flatten()(batch_X)), batch_y)
            err_cum += err
            loss_cum += loss * batch_y.shape[0]
            num_sample += batch_y.shape[0]
        return err_cum / num_sample, loss_cum / num_sample
    else:
        model.train()
        err_cum = 0
        loss_cum = 0
        num_sample = 0
        for batch in dataloader:
            opt.reset_grad()
            batch_X, batch_y = batch
            loss = nn.SoftmaxLoss()(model(nn.Flatten()(batch_X)), batch_y)
            loss.backward()
            _, err = loss_err(
                model(nn.Flatten()(batch_X)), batch_y)
            err_cum += err
            loss_cum += loss.numpy() * batch_y.shape[0]
            num_sample += batch_y.shape[0]
            opt.step()
        return err_cum / num_sample, loss_cum / num_sample

    # END YOUR SOLUTION


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    train_set = data.MNISTDataset(os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
                                  os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    test_set = data.MNISTDataset(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
                                 os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    test_loader = data.DataLoader(test_set, batch_size)
    model = MLPResNet(train_set.num_features, hidden_dim)
    opt = optimizer(model.parameters(), lr, weight_decay=weight_decay)
    for _ in range(epochs-1):
        train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
        epoch(train_loader, model, opt)
    train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
    train_err, train_loss = epoch(train_loader, model, opt)
    test_err, test_loss = epoch(test_loader, model, None)
    return train_err, train_loss, test_err, test_loss
    # END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")


def loss_err(h, y):
    """ Helper function to compute both loss and error"""
    return nn.SoftmaxLoss()(h.data, y.data).numpy(), np.sum(h.numpy().argmax(axis=1) != y.numpy())
