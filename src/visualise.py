import matplotlib.pyplot as plt
import numpy as np
from utils import rot_2d


def mr_data(X, y, ax=None):
    """
    populates axis with an MR image given by X and draws eyeball center and the line of sight based on y.

    Parameters
    ----------
    X : (43, 43) numpy array
        MR image with human eyeball
    y : (3,) numpy array
        containing column of eyeball center, row of eyeball center and eyeball orientation in this particular order
    ax: plt.axis handle, optional
        axis on which the image is plotted, default is current axis
    """

    # Assign current axis handle if none is given
    if ax is None:
        ax = plt.gca()

    # Calculate the line of sight based on eyeball center and orientation
    x0_lns = np.array([[y[0]], [y[1]]]) + rot_2d(y[2]) @ np.array([[0], [-1]])
    m = (x0_lns[0] - y[0]) / (x0_lns[1] - y[1])
    b0 = y[0] - m * y[1]

    # Plot MR image and
    ax.imshow(X, vmin=0, vmax=1.5)
    ax.scatter(y[0], y[1], s=150, c='k', marker='+', zorder=2)
    ax.plot(m * [0, X.shape[0]-1] + b0, [0, X.shape[1]-1], 'r', linewidth=1.5, zorder=1)


def model_loss(history):
    fig = plt.figure()

    plt.plot(history.history['loss'], label="loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.legend()
    plt.title("Can the CNN learn to predict eye position and orientation?")
    plt.xlabel("epoch")
    plt.ylabel("Loss")

    return fig


def test_performance(X, y_true, model):
    y_pred = model.predict(X)

    fig, axs = plt.subplots(2, 3)

    axs[0, 0].scatter(y_pred[:, 0], y_true[:, 0])
    axs[0, 1].scatter(y_pred[:, 1], y_true[:, 1])
    axs[0, 2].scatter(y_pred[:, 2], y_true[:, 2])
    axs[1, 0].hist(y_pred[:, 0] - y_true[:, 0])
    axs[1, 1].hist(y_pred[:, 1] - y_true[:, 1])
    axs[1, 2].hist(y_pred[:, 2] - y_true[:, 2])

    return fig


def mr_snapshot(axs, t, X, y):
    mr_data(X, y, ax=axs[0])
    axs[1].scatter(t, y[2])
