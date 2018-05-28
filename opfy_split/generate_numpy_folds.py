import os
import glob
import pickle
import logging

import numpy as np
import plac
from plac import annotations, Annotation

from opfy_split import utils

logging.basicConfig(level=logging.INFO)


def read_fold(fold_path):
    """Converts a text representation from OPF .dat to a a pickle.

    :param fold_path: The txt representing the OPF .dat file to be converted
    :returns (X, y) where X are the features and y the labels
    """

    labels = list()
    features = list()

    for number, line in enumerate(open(fold_path, 'r')):
        if number == 0:
            continue

        entries = line.split()
        label, vector = entries[1], entries[2:]

        labels.append(label)
        features.append(vector)

    labels = np.asarray(labels, dtype=np.int32)
    features = np.asarray(features, dtype=np.float32)

    return features, labels


def store_fold(features, labels, destination):
    """Stores numpy arrays as a Python pickle file.

    :param features: The X matrix
    :param labels: The y vector
    :param destination: Where the file will be created
    :returns None
    """

    with open(destination, 'wb') as file:
        data = {
            'X': features,
            'y': labels
        }

        pickle.dump(data, file)


def numpyfy_folder(opf_path, folder_path, destination):
    """Transforms all .dat files in a folder into corresponding pickle files.

    :param opf_path
    :param folder_path: Folder containing OPF .dat files
    :param destination: Where the generated pickle files will be stored
    :returns None
    """

    for fold in glob.glob(folder_path):
        filename = os.path.basename(fold)[:-4] + '.pyk'

        logging.info('Converting file {}'.format(fold))

        # converting OPF back to a text file
        utils.invoke(opf_path, '../tools/opf2txt', fold, 'temp.txt')

        # ... and from the text to a pickle file
        features, labels = read_fold('temp.txt')
        store_fold(features, labels, os.path.join(destination, filename))
        os.remove('temp.txt')


@annotations(
    opf_folder=Annotation('Destination folder used on generate_opf_folds',
                          type=str),
    numpy_folder=Annotation('Where the pickle folder files will be created. '
                            'Defaults to the argument opf_folder.', 'option',
                             type=str),
    opf_path=Annotation('Path to the OPF bin folder. Defaults to $OPF_PATH',
                        'option', type=str)
)
def main(opf_folder, numpy_folder=None, opf_path=None):
    """
    Converts OPF fold files, created using generate_opf_folds to corresponding
    numpy arrays persisted as pickle files.
    """


    if not numpy_folder:
        numpy_folder = opf_folder

    opf_path = utils.get_opf_path(opf_path)

    logging.info('[1/3] Creating target folders')
    utils.create_dir(os.path.join(numpy_folder, './numpy_cross'))
    utils.create_dir(os.path.join(numpy_folder, './numpy_folds'))

    logging.info('[2/3] Computing cross validation files.')
    numpyfy_folder(
        opf_path,
        os.path.join(opf_folder, 'cross_validation/*.dat'),
        os.path.join(numpy_folder, 'numpy_cross/')
    )

    logging.info('[3/3] Computing fold files.')
    numpyfy_folder(
        opf_path,
        os.path.join(opf_folder, 'folds/*.dat'),
        os.path.join(numpy_folder, 'numpy_folds/')
    )


def _entry_point():
    plac.call(main)
