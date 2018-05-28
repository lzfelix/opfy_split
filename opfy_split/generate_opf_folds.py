# coding:utf-8
import os
import glob
import logging

import plac
from plac import annotations, Annotation

from opfy_split import utils

logging.basicConfig(level=logging.INFO)
AMOUNT_FOLDS = 15


def create_train_partition(opf_path, cross_path, train_folds, fold_index,
                           dev_index, test_index):
    """Creates a single train partition using all available folds.

    :param opf_path
    :param cross_path: Where the generate .dat file will be stored
    :param train_folds: A list with the basename of all training folds
    :param fold_index: The index of this fold used to generate the new filename
    :param dev_index: The index of the corresponding dev fold
    :param test_index: The index of the corresponding test fold
    :returns None
    """

    # moving merged file to the cross_validation folder
    filenames = list(map(lambda x: '{}.dat'.format(x), train_folds))

    utils.invoke(opf_path, 'opf_merge', *filenames)

    # moving merged file to specific folder
    destination_filename = 'cross_{}_val_{}_test_{}.dat'.format(
        fold_index + 1, dev_index + 1, test_index + 1)

    # the merged file is created on this folder and then moved away
    destination_filename = os.path.join(cross_path, destination_filename)
    os.rename('merged.dat', destination_filename)


def store_folds(folds_path):
    for f in glob.glob('./fold*.dat'):
        destination_f = os.path.join(folds_path, os.path.basename(f))
        os.rename(f, destination_f)


@annotations(
    opf_samples_file=Annotation('Path to the OPF .dat file', type=str),
    amount_folds=Annotation('Amount of folds', 'option', 'k', int),
    destination=Annotation('Where to store the new .dat files. Defaults to the '
                           'samples file folder', type=str, kind='option'),
    opf_path=Annotation('Path to the OPF bin folder. Defaults to $OPF_PATH',
                        'option', type=str)
)
def main(opf_samples_file, amount_folds=AMOUNT_FOLDS, destination=None,
         opf_path=None):
    """
    Given a dataset on the OPF .dat file, divides it into multiple folds
    to use a cross-validation procedure using the opf_split program.
    """

    if not opf_path:
        logging.info('Using OPF path on the OPF_PATH env variable.')
        opf_path = os.environ['OPF_PATH']
    else:
        opf_path = os.path.expanduser(opf_path)

    print(opf_path)

    if not destination:
        logging.info('Saving on the same folder as the original .dat file.')
        destination = os.path.dirname(opf_samples_file)

    if amount_folds < 4:
        raise RuntimeError('k should be greater than 3.')

    # creating folders to accomodade the new files
    folds_path = os.path.join(destination, 'folds')
    utils.create_dir(folds_path)

    cross_path = os.path.join(destination, 'cross_validation')
    utils.create_dir(cross_path)

    # generating dataset folds
    utils.invoke(opf_path, 'opf_fold', opf_samples_file, amount_folds, 0)

    # generating the fold names. They are stored on this script folder (temp)
    folds = ['fold_{}'.format(i) for i in range(1, amount_folds+1)]

    # generating the folds

    for fold_index in range(amount_folds):
        logging.info('Generating cross-validation partition {}/{}'.format(
            fold_index + 1,
            amount_folds
        ))

        # training_folds = {all_folds} - {val_fold, test_fold}
        dev_index = fold_index
        test_index = (fold_index + 1) % amount_folds

        train_folds = list(folds)
        train_folds.remove(folds[dev_index])
        train_folds.remove(folds[test_index])

        create_train_partition(opf_path, cross_path, train_folds, fold_index,
                               dev_index, test_index)

    store_folds(folds_path)


def _entry_point():
    plac.call(main)
