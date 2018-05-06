import os
import pickle

import numpy as np


def _restore_files(filepath):
    """Given a fold file, returns its X matrix, y vector and the filepath."""
    
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
        X, y = data['X'], data['y']
        
    return X, y, filepath


def _read_pickle(folds_folder, filepath, index):
    """Restore the dev/test fold for a given cross-validation split.
    
    > folds_folder: Path to the folder containing all folds.
    > filepath: The cross-validation file.
    > index: The index that determines the test/dev fold id.
    < X, y and the fold filename
    """
    
    # keep just the filename (ignore possible extension) and identify
    # the corresponding test/dev split id
    filepath = os.path.basename(filepath)
    index = filepath.split('_')[index]
    index = index.split('.')[0]
    
    # restoring the fold filename for numpy
    filename = 'fold_{}.pyk'.format(index)
    
    # load the features and labels
    return _restore_files(os.path.join(folds_folder, filename))


def get_dev_fold(folds_folder, filepath):
    """Restores the dev X matrix and y vector for a cross-validation filename.
    
    > folds_folder: Path to the folder containing all folds.
    > filepath: Cross-validation split filename.
    < X, y, filename of the corresponding dev fold.
    """
    return _read_pickle(folds_folder, filepath, 3)


def get_test_fold(folds_folder, filepath):
    """Restores the test X matrix and y vector for a cross-validation filename.
    
    > folds_folder: Path to the folder containing all folds.
    > filepath: Cross-validation split filename.
    < X, y, filename of the corresponding dev fold.
    """
    return _read_pickle(folds_folder, filepath, 5)


def fetch_folds(path):
    """Returns a generator that at each step restores one cross-validation fold.
    > path: Path to the folder with all folds for cross-validating.
    < X, y, filename of the corresponding training fold.
    """
    files = sorted(os.listdir(path))
    for file in files:
        yield _restore_files(os.path.join(path, file))

