# opfy_split

A set of scripts that aim helping users to create cross-validation folds from
OPF data files and to map these files to corresponding numpy arrays, thus
allowing classifiers implemented in Python to be directly compared to OPF by
running both techniques on equal folds.

This utility fills the gap between the OPF framework, which is implemented in C,
and the classifiers implemented in Python. This divergence exists because the
OPF uses files with the `.dat` characteristic extension to save the data to be
parsed.

The scripts can be used on two ways:

* To simply create OPF cross-validation folds using a simple interface
* To map the OPF folds to their corresponding counter parts, allowing direct
comparison between Python-implemented and OPF classifier.


## Instructions

Since this is a simple wrapper around some of the OPF utility programs, namely
`opf_split`, `opf_merge` and `opf2txt`. First it's necessary to download,
compile LibOPF and optionally create an environment variable pointing to its
`bin/` folder named `$OPF_PATH`:

```bash
git clone https://github.com/jppbsi/LibOPF
make
export PATH=$(pwd)/bin
```

Currently, the only way to install `opfy_split` is by cloning the repository.
You might want to do the following steps inside a virtual environment:

```bash
git clone https://github.com/lzfelix/opfy_split
pip install .
```

Following two shell-like commands will be available: `opfy_opf` and
`opfy_numpy`. The first generates `k` folds given an OPF dataset on the `.dat`
format and the latter converts the generated folds to their correponding numpy
representation.


## Example

Consider the following code that generates 4 folds for the cone-toruns dataset,
available along LibOPF:

```bash
mkdir example; cd example/              # first create an isolated folder
cp $OPF_PATH/../data/cone-torus.dat .   # copy the dataset of interest

opfy_opf cone-torus.dat -k=4            # generate the OPF folds
# INFO:root:Saving on the same folder as the original .dat file.
# INFO:root:Generating cross-validation partition 1/7
# INFO:root:Generating cross-validation partition 2/7
# INFO:root:Generating cross-validation partition 3/7
# INFO:root:Generating cross-validation partition 4/7
# INFO:root:Generating cross-validation partition 5/7
# INFO:root:Generating cross-validation partition 6/7
# INFO:root:Generating cross-validation partition 7/7

opfy_numpy .                            # create the corresponding numpy files

tree
# .
# ├── cone-torus.dat
# ├── cross_validation
# │   ├── cross_1_val_1_test_2.dat
# │   ├── cross_2_val_2_test_3.dat
# │   ├── cross_3_val_3_test_4.dat
# │   └── cross_4_val_4_test_1.dat
# ├── folds
# │   ├── fold_1.dat
# │   ├── fold_2.dat
# │   ├── fold_3.dat
# │   └── fold_4.dat
# ├── numpy_cross
# │   ├── cross_1_val_1_test_2.pyk
# │   ├── cross_2_val_2_test_3.pyk
# │   ├── cross_3_val_3_test_4.pyk
# │   └── cross_4_val_4_test_1.pyk
# └── numpy_folds
#     ├── fold_1.pyk
#     ├── fold_2.pyk
#     ├── fold_3.pyk
#     └── fold_4.pyk
```

The numbers on the cross-validation filenames correspond, respectivelly, to the
number of the cross-validation fold, the corresponding validation fold and the
corresponding test fold. Notice that although the first two values are equal,
the validation fold **is not** on the training partition. For instance, if you
train a classifier with `cross_3_val_3_test_4.dat`, the you should evaluate it
on `fold_3` and test it on `fold_4`.


## Command line utility arguments

The utility scripts can be invoked with the `-h` flag to provide some help
instructions, but basically it is possible to pass as argument the `opf_path`
variable, thus making the use of `$OPF_PATH` optional, to set destination folder
for the created files and to set the amount of folds to be created (for
`opfy_opf` only).


## Training a Python classifier

After having installed the opfy_split library, it can also be imported on the
code to automatically iterate over the `numpy_` folders, restoring the necessary
folds to train/evaluate and test a classifier:

```python
from opf_split import opf_split
from sklearn.metrics import accuracy_score
from sklearn.classifiers import SomeClassifier

cross_dir = './example/numpy_cross/'
folds_dir = './example/numpy_folds/'

for fold in opf_split.fetch_folds(cross_dir):
    X, y, cross_val_file = fold

    clf = SomeClassifier()
    predictions_train = clf.fit_predict(X, y)

    print('Train acc: {}'.format(accuracy_score(y, predictions_train)))

    X, y, _ = opf_split.get_dev_fold(folds_dir, cross_val_file)
    predictions_dev = clf.fit_predict(X, y

    print('Dev acc: {} '.format(accuracy_score(y, predictions_dev)))
```


## Notice

This scripts invokes underlying OPF programs, namely `opf_split`, `opf_merge`
and `opf2txt` using the Python module `subprocess`. It was tested on Linux and
OSX systems, but not on Windows (this does not necessarily means that it
doesn't work on this platform).
