import numpy as np


def rot_2d(theta):
    """
    2D rotation matrix (counter-clockwise).

    Parameters
    ----------
    theta : float
        amount of counter-clockwise rotation in degrees.

    Returns
    -------
    numpy.ndarray
        2D numpy rotation matrix
    """

    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)

    return np.array([[c, s], [-s, c]])


def in_ellipse(x, elp_center, elp_radius, elp_rotation):
    """
    tests whether coordinates x lie inside an ellipse by projecting it onto a unit circle via affine
    transformation.

    Parameters
    ----------
    x : numpy.ndarray
        coordinate in question
    elp_center: numpy.ndarray
        center of the ellipse
    elp_radius: numpy.ndarray
        radii of the two semi-axes of the ellipse
    elp_rotation: numpy.ndarray
        orientation of the ellipse.

    Returns
    -------
    numpy.ndarray
        booleans of the same size as x, depending on whether each coordinate is in- or outside the ellipse.
    """

    x_projection = np.linalg.norm(np.diag(1.0 / elp_radius) @ elp_rotation.T @ (x - elp_center), axis=0)

    return x_projection < 1
