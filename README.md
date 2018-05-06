# OPF Helpers

**NOTICE: This is a work-in-progress repo.**

This folder contains helper scripts to generate the cross-validation folds for training,
evaluation and test of a dataset using both OPF and other Python-based classifiers. The
need for these scripts steems from two aspects, discussed below.

The OPF classifier is currently implemented in C, as well as its main functionalities to 
pre-compute distances between samples and folds, which makes hard to do these calculations 
in in Python and then map them to the OPF format, while keeping sync amongst files.

On the other hand, we would like to apply the Wilcoxon Signed Rank test between the 
evaluations in order to statistically determine if two different classifiers produce 
statistically significant different results. In order to do so the folds must be equal 
across experiments.

Hence, these scripts can be used to: given the OPF pre-computed distances (which speeds up
the classifier performance), compute the OPF folds and then map the file back to a numpy
representation, allowing to use the very same folds across both languages, and consequently,
classifiers. The steps are the following:

```
    1. export and enviroment variable $OPF_PATH containing the path to the OPF binary folder,
    ie: /home/user/LibOPF/bin/

    2. generate a text file on the OPF format with the samples to be used for training, 
    evaluation and test. Following use $OPF_PATH/../tools/opf2txt filename.txt filename.dat.
    Please notice that this file should contain label information if it is desired to benchmark
    unsupervised OPF with supervised metrics, such as accuracy.

    3. pre-compute the distances between samples with the $OPF_PATH/opf_distance command. This
    will generate a distances.dat file.

    3.1. if the distance files were generated in another folder, instead of copying them to this
    folder, it's possible to create a symbolic link instead.
    
    4. to generate the folds, use python generate_all_folds.py. This script is currently hardcoded
    to expect the samples file having the name all_samples.dat and it will create and populate
    two folders: folds/ and cross_validation/

    The files on the cross_validation/ folder have the following name convention:
    <cross_N_val_M_test_O.pyk>, where N is the fold number, M is the number of the fold which should
    be used for validation and O the number of the corresponding testing fold. Note that although 
    N == M usually, the Mth fold is *not* included on the Nth cross-validation training fold. This is
    just how the naming convention works.

    5. from these files it's possible to generate the very same numpy representation. To do so, simple
    use python generate_numpy_folds.py. This script will read the folder structure created by the 
    previous program and generate two more folders, numpy_folds and numpy_cross with the same data as
    their counterparts, but with numpy-compatible files.
```

## How to use

After having generated the numpy version of the OPF splits, it's possible to loaad them to
train a classifier

```python
    from opf_split import opf_split
    from sklearn.metrics import accuracy_score

    cross_dir = '...'
    folds_dir = '...'

    for fold in opf_split.fetch_folds(cross_dir):
        X, y, cross_val_file = fold

        clf = SomeClassifier()
        predictions_train = clf.fit_predict(X, y)

        print('Train acc: {}'.format(accuracy_score(y, predictions_train)))

        X, y, _ = opf_split.get_dev_fold(folds_dir, cross_val_file)
        predictions_dev = clf.fit_predict(X, y

        print('Dev acc: {} '.format(accuracy_score(y, predictions_dev)))

```

## Notes

* These scripts require `Python 3.x` and `numpy` and `plac` libraries.
* Currently these scripts rely on OS features and are only UNIX-compatible.
* OPF labels are 1-indexed, ie: the classes indices start at 1, while on sklearn they are 0-indexed.
The provided scripts don't modify this scheme.
* It's also possible to install this lib with `pip install .`, although this is still an 
experimental feature

