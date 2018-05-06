import os
import sys
import subprocess

import plac

AMOUNT_FOLDS = 15


def insert_dat(x):
    return '{}.dat'.format(x)


def main(opf_samples_file, amount_folds=AMOUNT_FOLDS, destination=None):

    # if destination is not provided, save on the same location
    if not destination:
        destination = os.path.dirname(opf_samples_file)

    # OPF utilities will create all the new files here
    origin_path = os.path.dirname(opf_samples_file)

    # creating folders to accomodade the new files
    folds_path = os.path.join(destination, 'folds')
    if not os.path.exists(folds_path):
        os.mkdir(folds_path)

    cross_path = os.path.join(destination, 'cross_validation')
    if not os.path.isdir(cross_path):
        os.mkdir(cross_path)

    # fixing type conversion for plac
    if type(amount_folds) is str:
        amount_folds = int(amount_folds)

    # generating dataset folds
    os.system('$OPF_PATH/opf_fold {} {} 0'.format(opf_samples_file, amount_folds))

    # generating the fold names. They are stored on this script folder (temp)
    folds = ['fold_{}'.format(i) for i in range(1, amount_folds+1)]

    # generating the folds
    for i in range(len(folds)):

        # validation = all_folds - {val_fold, test_fold}
        validation_index = i
        test_index = (i + 1) % amount_folds

        train_folds = list(folds)
        train_folds.remove(folds[validation_index])
        train_folds.remove(folds[test_index])

        # moving merged to the cross_validation folder
        filenames = list(map(insert_dat, train_folds))
        os.system('$OPF_PATH/opf_merge {}'.format(' '.join(filenames)))
        print('$OPF_PATH/opf_merge {}'.format(' '.join(filenames)))

        # moving merged file to specific folder
        destination_filename = 'cross_{}_val_{}_test_{}.dat'.format(
            i + 1, validation_index + 1, test_index + 1)

        # the merged file is created on this folder
        destination_filename = os.path.join(cross_path, destination_filename)
        os.rename('merged.dat', destination_filename)

        # some logging
        print('Executed> $OPF_PATH/opf_merge {}'.format(' '.join(filenames)))
        print('Generating {}'.format(destination_filename))

    # moving fold files to specific folder
    os.system('mv ./fold* {}'.format(folds_path))


if __name__ == '__main__':
    plac.call(main)

