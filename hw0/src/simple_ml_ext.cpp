#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;
/**
 * A handy way of writing C = np.matmul(A, B), where A is in (m*n),
 * B is in (n*k) and C is in (m*k).
 * If transposeA is true, then A should be A^T.
 */
#define MATMUL(A, B, C, m, n, k, transposeA)                                                    \
    do                                                                                          \
    {                                                                                           \
        for (size_t p = 0; p < (m); p++)                                                        \
            \                             
                                                                                                                                                                               \
            {                                                                                   \
                for (size_t q = 0; q < (k); q++)                                                \
                {                                                                               \
                    (C)[p * (k) + q] = 0;                                                       \
                    for (size_t r = 0; r < (n); r++)                                            \
                        (C)[p * (k) + q] +=                                                     \
                            (A)[transposeA ? (r * (m) + p) : (p * (n) + r)] * (B)[r * (k) + q]; \
                }                                                                               \
            }                                                                                   \
    } while (0)

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t num_batches = (m + batch) / batch;
    for (size_t i = 0; i < num_batches; i++)
    {
        size_t batch_size = (m < (i + 1) * batch) ? (m - i * batch) : batch;

        float *Z = new float[batch_size * k],
              *grad = new float[n * k];
        const float
            *X_selected = X + i * batch * n;
        const unsigned char
            *y_selected = y + i * batch;
        MATMUL(X_selected, theta, Z, batch_size, n, k, false);
        // normalize Z and minus one-hot index
        for (size_t p = 0; p < batch_size; p++)
        {
            float store = 0;
            for (size_t q = 0; q < k; q++)
            {
                Z[p * k + q] = std::exp(Z[p * k + q]);
                store += Z[p * k + q];
            }
            for (size_t q = 0; q < k; q++)
            {
                Z[p * k + q] /= store;
                if (y_selected[p] == q)
                {
                    Z[p * k + q] -= 1;
                }
                Z[p * k + q] /= batch_size;
            }
        }
        MATMUL(X_selected, Z, grad, n, batch_size, k, true);
        for (size_t p = 0; p < n; p++)
        {
            for (size_t q = 0; q < k; q++)
            {
                theta[p * k + q] -= lr * grad[p * k + q];
            }
        }
        delete[] Z;
        delete[] grad;
    }
    /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m)
{
    m.def(
        "softmax_regression_epoch_cpp",
        [](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch)
        {
            softmax_regression_epoch_cpp(
                static_cast<const float *>(X.request().ptr),
                static_cast<const unsigned char *>(y.request().ptr),
                static_cast<float *>(theta.request().ptr),
                X.request().shape[0],
                X.request().shape[1],
                theta.request().shape[1],
                lr,
                batch);
        },
        py::arg("X"), py::arg("y"), py::arg("theta"),
        py::arg("lr"), py::arg("batch"));
}
