import matplotlib.pyplot as plt
import numpy as np
import imageio
from utils import rot_2d


def mr_data(X, y=None, ax=None):
    """
    populates axis with an MR image given by X and draws eyeball center and the line of sight based on y.

    Parameters
    ----------
    X : (43, 43) numpy array
        MR image with human eyeball
    y : (3,) numpy array, optional
        containing column of eyeball center, row of eyeball center and eyeball orientation in this particular order
    ax: plt.axis handle, optional
        axis on which the image is plotted, default is current axis
    """

    # Assign current axis handle if none is given
    if ax is None:
        ax = plt.gca()

    # Plot MR image
    ax.imshow(X, cmap='gray', vmin=0, vmax=1.5)

    # If eyeball kinematics are given, calculate line of sight and plot on top of MR data
    if y is not None:
        # Calculate the line of sight based on eyeball center and orientation
        x0_lns = np.array([[y[0]], [y[1]]]) + rot_2d(y[2]) @ np.array([[0], [-1]])
        m = (x0_lns[0] - y[0]) / (x0_lns[1] - y[1])
        b0 = y[0] - m * y[1]

        # Plot eyeball center and line of sight
        ax.scatter(y[0], y[1], s=150, c='k', marker='+', zorder=2)
        ax.plot(m * [0, X.shape[0]-1] + b0, [0, X.shape[1]-1], 'r', linewidth=1.5, zorder=1)


def model_loss(history):
    """
    plots training and test set loss over epochs from the model fitting process. Takes the model history as input and
    returns the figure handle.
    """

    # Initiate Figure and axes handle
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # plot loss over epochs, but exclude the very first epoch
    ax.plot(history.history['loss'][1:], label="loss")
    ax.plot(history.history['val_loss'][1:], label="val_loss")
    ax.legend()
    ax.set_title("MSE loss from predicting eye kinematics in artificial MR data")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Loss")

    return fig


def test_set_performance(X, y_true, model):
    """
    plots the model performance on data set X by comparing it to the true outputs y.

    Parameters
    ----------
    X : (n, 43, 43) numpy array
        dataset of n MR images of human eyeballs
    y_true : (n, 3) numpy array
        containing true column & row of eyeball center and eyeball orientation of each image
    model: tensorflow model
        used to predict eye kinematics from X and compare to y_true
    """

    # Predict eyeball kinematics with model
    y_pred = model.predict(X)

    # Prepare figure axes
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 9))
    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    axs[0, 0].set_xlabel('True column of eyeball center [px]')
    axs[0, 0].set_ylabel('Predicted column of eyeball center [px]')
    axs[0, 0].set_xlim(17, 25)
    axs[0, 0].set_ylim(17, 25)
    axs[0, 1].set_xlabel('True row of eyeball center [px]')
    axs[0, 1].set_ylabel('Predicted row of eyeball center [px]')
    axs[0, 1].set_xlim(17, 25)
    axs[0, 1].set_ylim(17, 25)
    axs[0, 2].set_xlabel('True eyeball orientation [°]')
    axs[0, 2].set_ylabel('Predicted eyeball orientation [°]')
    axs[0, 2].set_xlim(-25, 25)
    axs[0, 2].set_ylim(-25, 25)
    axs[1, 0].set_xlabel('Difference between ground truth and prediction [px]')
    axs[1, 0].set_ylabel('Count [ ]')
    axs[1, 0].set_xlim(-1.5, 1.5)
    axs[1, 1].set_xlabel('Difference between ground truth and prediction [px]')
    axs[1, 1].set_ylabel('Count [ ]')
    axs[1, 1].set_xlim(-1.5, 1.5)
    axs[1, 2].set_xlabel('Difference between ground truth and prediction [°]')
    axs[1, 2].set_ylabel('Count [ ]')
    axs[1, 2].set_xlim(-8, 8)

    # Subplot 1-3: Scatter plot of true vs predicted eye kinematics
    for i in range(3):
        axs[0, i].scatter(y_true[:, i], y_pred[:, i], s=4, color='gray')
        axs[0, i].plot(axs[0, i].get_xlim(), axs[0, i].get_xlim(), color='black')

    # Subplot 4-6: Histogram of difference between true and predicted eye kinematics plus mean and std
    for i in range(3):
        axs[1, i].hist(y_pred[:, i] - y_true[:, i], facecolor='gray', edgecolor='black')
        x_lim = axs[1, i].get_xlim()
        y_lim = axs[1, i].get_ylim()
        mu = np.mean(y_pred[:, i] - y_true[:, i])
        sigma = np.std(y_pred[:, i] - y_true[:, i])
        axs[1, i].text(x_lim[0] + 0.05 * (x_lim[1] - x_lim[0]),
                       y_lim[0] + 0.90 * (y_lim[1] - y_lim[0]),
                       r'$\mu={:.2f}$'.format(mu),
                       fontsize=12)
        axs[1, i].text(x_lim[0] + 0.05 * (x_lim[1] - x_lim[0]),
                       y_lim[0] + 0.80 * (y_lim[1] - y_lim[0]),
                       r'$\sigma={:.2f}$'.format(sigma),
                       fontsize=12)

    return fig


def mr_snapshot_and_time_series(frame, t, X, y, axs):
    """
    Plots the MRI data of the current frame along with the time series of eyeball orientation up onto that particular
    frame.

    Parameters
    ----------
    frame: int
        current frame out of the n frames available
    t : (n, ) numpy array
        time array for all frames
    X : (n, 43, 43) numpy array
        dataset of n MR images of human eyeballs
    y : (n, 3) numpy array
        containing column & row of eyeball center and eyeball orientation of each image
    axs : (2,) axis handle
    """

    # Clear previous frame
    axs[0].clear()
    axs[1].clear()

    # Keep axes  constant across all frames
    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)
    axs[1].set_xlim(np.min(t), np.max(t))
    axs[1].set_ylim(-15, 10)
    axs[1].set_xlabel('Time [s]', fontsize=12)
    axs[1].set_ylabel('Gaze Angle [°]', fontsize=12)

    # Plot MR data and time series
    mr_data(X[frame, :, :], y[frame, :], ax=axs[0])
    axs[1].plot(t[0:(frame + 1)], y[0:frame + 1, 2], c='r')
    axs[1].scatter(t[frame], y[frame, 2], c='r')

    # draw the canvas, cache the renderer
    fig = plt.gcf()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image


def create_gif(X, y):
    """
    Create a gif which displays the MRI data in X along with eyeball kinematics estimation given by y, side-by-side
    with the time series of the gaze angle

    X : (n, 43, 43) numpy array
        dataset of n MR images of human eyeballs
    y : (n, 3) numpy array
        containing column & row of eyeball center and eyeball orientation of each image
    """

    # pick the frame range for the gif
    idx = np.arange(0, 500)
    t = idx * 0.055
    X = X[idx, :, :]
    y = y[idx, :]

    # create figure and axes handle
    fig = plt.figure(figsize=(12, 3))
    ax1 = fig.add_axes([0, 0, 0.25, 1])
    ax2 = fig.add_axes([0.34, 0.19, 0.64, 0.66])
    axs = [ax1, ax2]

    # create gif py plotting frame-by-frame
    imageio.mimsave('../results/eye_movement.gif',
                    [mr_snapshot_and_time_series(i, t, X, y, axs) for i in range(len(idx))],
                    fps=18)
