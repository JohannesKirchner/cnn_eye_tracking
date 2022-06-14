import numpy as np
import cv2
import h5py
from utils import rot_2d, in_ellipse
from sklearn.model_selection import train_test_split


def generate_artificial_data(n=40000):
    """
    Generates a set of n artificial MRI images with randomised eye position, rotation, radii etc.
    """

    # Parameter determining spread in eyeball radii, image intensity and relative angle between cornea & lens
    radius_std = 0.3
    intensity_std = 0.1
    angle_std = 2.0

    # Create artificial MRI images of human eyes with randomised location and orientation
    X = np.zeros((n, 43, 43))
    y = np.zeros((n, 3))
    for i in range(n):
        if i % 100 == 0:
            print('{}/{}'.format((i // 100) + 1, (n // 100)))
        y[i, 0] = 21 + 6 * (np.random.rand() - 0.5)
        y[i, 1] = 21 + 6 * (np.random.rand() - 0.5)
        y[i, 2] = 0 + 40 * (np.random.rand() - 0.5)
        X[i, :, :] = artificial_mri_image(y[i, 0], y[i, 1], y[i, 2], radius_std=radius_std,
                                          intensity_std=intensity_std, angle_std=angle_std)

    # Split data into train and test set, then save everything
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    with h5py.File('../data/artificial_MRI_data.h5', 'w') as f:
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('y_train', data=y_train)
        f.create_dataset('X_test', data=X_test)
        f.create_dataset('y_test', data=y_test)


def artificial_mri_image(center_x, center_y, angle, radius_std=0.2, intensity_std=0.1, angle_std=2.0):
    """
    returns an artificial mri image of a human eyeball with location and orientation specified in parameters

    Parameters
    ----------
    center_x : float
        column of eyeball center location
    center_y: float
        row of eyeball center location
    angle: float
        eyeball orientation in degree
    radius_std: float, optional
        standard deviation of the normally distributed radii of sclera, lens & cornea
    intensity_std: float, optional
        standard deviation of the normally distributed image intensities
    angle_std: float, optional
        standard deviation of the normally distributed relative rotation between sclera and lens & cornea in degrees

    Returns
    -------
    numpy.ndarray
        43 x 43 numpy array of an artificial MRI image of a human eyeball with respective parameters
    """

    # Basic image parameter: size, granularity & background intensity
    imsz = (43, 43)
    g = 4
    background_intensity = 0.4

    # Sclera parameter
    x0_scl = np.array([[center_x], [center_y]])
    rad_scl = np.array((13.3, 12.2)) + np.random.normal(0, radius_std, 2)
    rot_scl = rot_2d(np.random.normal(angle, angle_std))

    # Cornea parameter
    rot_crn = rot_2d(np.random.normal(angle, angle_std))
    x0_crn = x0_scl + 7.8 * rot_crn @ np.array([[0], [-1]])
    rad_crn = np.array((7.8, 8.2)) + np.random.normal(0, radius_std, 2)
    crn_intensity = np.random.normal(1.4, intensity_std)

    # Lens parameter
    rad_lns = np.array((6.3, 5.3)) + np.random.normal(0, radius_std, 2)
    rot_lns = rot_2d(angle)
    lam = 1 / np.linalg.norm(np.diag(1.0 / rad_scl) @ rot_scl.T @ rot_lns @ np.array([[0], [-1]]))
    x0_lns = x0_scl + lam * rot_lns @ np.array([[0], [-1]])

    # Preallocate image with randomised noise
    back_x = np.linspace(np.random.rand(), np.random.rand(), imsz[0])
    back_y = np.linspace(np.random.rand(), np.random.rand(), imsz[1])
    back_X, back_Y = np.meshgrid(back_x, back_y)
    img = np.multiply(np.abs(np.random.normal(0, background_intensity, imsz)), np.add(back_X, back_Y))

    # Loop over grid points for each pixel and check in which eye structure they fall (if any)
    Y, X = np.mgrid[-0.5:(0.5 + 1 / (g - 1)):(1 / (g - 1)), -0.5:(0.5 + 1 / (g - 1)):(1 / (g - 1))]
    for i in range(imsz[0]):
        for j in range(imsz[1]):
            coord = [[i], [j]] + np.vstack([X.ravel(), Y.ravel()])

            # masking grid points inside the eyeball
            mask_scl = in_ellipse(coord, x0_scl, rad_scl, rot_scl) | in_ellipse(coord, x0_crn, rad_crn, rot_crn)

            # masking grid points inside the cornea
            mask_crn = in_ellipse(coord, x0_crn, rad_crn, rot_crn) & in_ellipse(coord, x0_lns, rad_lns - 2.7, rot_lns)

            # masking grid points inside the lens
            mask_lns = in_ellipse(coord, x0_lns, rad_lns, rot_lns) & in_ellipse(coord, x0_scl, rad_scl - 1.2, rot_scl)

            # masking grid points inside the eyeball but not the lens
            mask_eye = np.mean(mask_scl & ~mask_lns)

            # assign specific intensity depending on eye structure
            if mask_eye > 0:
                if np.sum(mask_crn) > 0:
                    img[j, i] = np.random.normal(crn_intensity * mask_eye + (1 - mask_eye) * img[j, i], intensity_std)
                else:
                    img[j, i] = np.random.normal(mask_eye + (1 - mask_eye) * img[j, i], intensity_std)
            elif np.sum(mask_scl) > 0:
                img[j, i] = 0

    # Blur the image with a gaussian filer
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


if __name__ == "__main__":
    generate_artificial_data()
