# coding:utf-8
import os
import pickle


def _restore_files(filepath):
    """
    Given a fold pickle file, returns its correponding X matrix, y vector and
    filepath argument.

    :param filepath: Path to the pickle representation of an OPF fold .dat file
    :returns: A tuple (X, y, filepath).
    """

    with open(filepath, 'rb') as file:
        data = pickle.load(file)
        X, y = data['X'], data['y']

    return X, y, filepath


def _read_pickle(folds_folder, filepath, index):
    """
    Restore either the correponding pickle file corresponding to the dev or
    test related to a given cross-validation fold.

    :param folds_folder: Path to the folder containing all the pickle objects
    corresponding to OPF .dat files.
    :param filepath: The cross-validation pickle filename.
    :param index: The index that determines which number from the filepath
    should be used to restore the desired data partition.
    :returns: A tuple (X, y, f), where f is the filename of the restored pickle
    file from the disk.
    """

    # keep just the filename (ignore possible extension) and identify the
    # corresponding test/dev split id
    filepath = os.path.basename(filepath)
    index = filepath.split('_')[index]
    index = index.split('.')[0]

    # restoring the fold filename for numpy and loading features and labels
    filename = 'fold_{}.pyk'.format(index)
    return _restore_files(os.path.join(folds_folder, filename))


def get_dev_fold(folds_folder, filepath):
    """
    Restores the corresponding dev X, y components of a cross-validation fold
    from a pickle file.

    :param folds_folder: Path to the folder containing all the pickle objects
    corresponding to OPF .dat files.
    :param filepath: The cross-validation pickle filename.
    :returns: A tuple (X, y, f), where f is the filename of the restored pickle
    file.
    """
    return _read_pickle(folds_folder, filepath, 3)


def get_test_fold(folds_folder, filepath):
    """
    Restores the corresponding test X, y components of a cross-validation
    fold from a pickle file.

    :param folds_folder: Path to the folder containing all the pickle objects
    corresponding to OPF .dat files.
    :param filepath: The cross-validation pickle filename.
    :returns: A tuple (X, y, f), where f is the filename of the restored pickle
    file.
    """
    return _read_pickle(folds_folder, filepath, 5)


def fetch_folds(path):
    """
    Returns a generator that, at each step, restores one cross-validation
    fold from the pickle files on the disk.

    :param path: Path to the folder with all folds for cross-validation pickle
    files.
    :returns A tuple (X, y, f) of the corresponding training part of the cross-
    validation fold, where f is the name of the restored file.
    """
    files = sorted(os.listdir(path))
    for file in files:
        yield _restore_files(os.path.join(path, file))
