import numpy as np
import gzip
import struct
import sys
sys.path.append('python/')
try:
    import needle as ndl
    import needle.ops as ops
except:
    pass


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    # BEGIN YOUR CODE
    with gzip.open(image_filename) as f:
        content = f.read()
        magic, length, row, column = struct.unpack(">iiii", content[:16])
        X = struct.unpack(">"+"B"*length*row*column, content[16:])
        X = np.array(X, dtype=np.float32).reshape(length, row*column)
        X = X/X.max()

    with gzip.open(label_filename) as f:
        content = f.read()
        magic, length = struct.unpack(">ii", content[:8])
        y = struct.unpack(">"+"B"*length, content[8:])
        y = np.array(y, dtype=np.uint8)

    return (X, y)

    # END YOUR CODE


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    # BEGIN YOUR SOLUTION
    batch_size = Z.shape[0]
    return ops.summation(ops.log(ops.summation(ops.exp(Z), axes=(1, )))
                         - ops.summation(Z * y_one_hot, axes=(1, ))) / batch_size
    # END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    # BEGIN YOUR SOLUTION
    num_batches = int(np.ceil(y.shape[0] / batch))
    num_classes = W2.shape[1]

    for i in range(num_batches):
        batch_size = batch
        # the last batch
        if y.shape[0] < (i+1)*batch:
            batch_size = y.shape[0] - i*batch

        # X_selected is in (batch_size x input_dim)
        # y_selected is in (batch_size)
        X_selected = X[i*batch:i*batch+batch_size]
        y_selected = y[i*batch:i*batch+batch_size]

        # convert y_selected to one-hot matrix
        mask = np.zeros((batch_size, num_classes))
        mask[np.arange(batch_size), y_selected] = 1
        mask = ndl.Tensor(mask, requires_grad=False)

        X_selected = ndl.Tensor(X_selected)
        loss = softmax_loss(ops.matmul(
            ops.relu(ops.matmul(X_selected, W1)), W2), mask)
        loss.backward()
        W1_numpy, W2_numpy = W1.numpy(), W2.numpy()
        W1_grad, W2_grad = W1.grad.numpy(), W2.grad.numpy()
        # update parameters
        W1_numpy -= W1_grad*lr
        W2_numpy -= W2_grad*lr

        W1.data = ndl.Tensor(W1_numpy)
        W2.data = ndl.Tensor(W2_numpy)

    return W1, W2
    # END YOUR SOLUTION

# CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
