import os
import glob
import pickle

from os import path

import plac
import numpy as np

print('Convert OPF fold files to numpy matrices')


def read_fold(fold_path):
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
    with open(destination, 'wb') as file:
        data = {
            'X': features,
            'y': labels
        }

        pickle.dump(data, file)


def numpyfy_folder(folder_path, destination):
    for fold in glob.glob(folder_path):
        filename = os.path.basename(fold)[:-4] + '.pyk'

        print('Analysing file {}'.format(fold))

        os.system('$OPF_PATH/../tools/opf2txt {} {}'.format(
            fold, 'temp.txt'
        ))

        features, labels = read_fold('temp.txt')
        store_fold(features, labels, os.path.join(destination, filename))
        os.remove('temp.txt')


def create_dir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)


def main(opf_folder, numpy_folder=None):

    if not numpy_folder:
        numpy_folder = opf_folder

    create_dir(path.join(opf_folder, './numpy_cross'))
    create_dir(path.join(opf_folder, './numpy_folds'))

    print('Computing cross validation files.')
    numpyfy_folder(
        path.join(opf_folder, 'cross_validation/*.dat'), 
        path.join(numpy_folder, 'numpy_cross/')
    )

    print('Computing fold files.')
    numpyfy_folder(
        path.join(opf_folder, 'folds/*.dat'), 
        path.join(numpy_folder, 'numpy_folds/')
    )


if __name__ == '__main__':
    plac.call(main)

