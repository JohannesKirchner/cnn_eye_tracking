import h5py
from architecture import cnn_model
import visualise


def train_model(epochs=10, plot=False):
    """
    Training of the CNN model with artificial MRI data. The complete model including trained weights is saved.

    Parameters
    ----------
    epochs : int, optional
        Number of epochs over which the model is fitted (default is 15).
    plot: bool, optional
        Choose whether to plot and save the model loss over epochs and the model performance on the artificial test
        set (default is False).
    """

    # Load train & test set from artificial MRI data
    with h5py.File('../data/artificial_MRI_data.h5', 'r') as f:
        X_train = f['X_train'][:]
        y_train = f['y_train'][:]
        X_test = f['X_test'][:]
        y_test = f['y_test'][:]

    # Create instance of the CNN model defined in architecture
    model = cnn_model()

    # Fit model with artificial training data and save the fit history
    history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))

    # Save trained model and weights to hd5 file
    model.save('../model/eye_tracking_cnn_model.h5')

    if plot:
        # Print & save loss function over epochs
        fig = visualise.model_loss(history)
        fig.savefig('../results/model_loss_over_epochs.png', bbox_inches='tight')

        # Print & save model performance on the test set
        fig = visualise.test_set_performance(X_test, y_test, model)
        fig.savefig('../results/test_set_performance.png', bbox_inches='tight')


if __name__ == "__main__":
    train_model(15, plot=True)
