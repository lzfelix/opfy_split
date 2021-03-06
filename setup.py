from setuptools import setup, find_packages


setup(
    name='opfy_split',
    version='0.2',
    author='Luiz Felix',
    author_email='lzcfelix@gmail.com',
    description=('Utility to sync numpy and OPF folds to train, evaluate and '
                 'test classifiers.'),
    python_requires='>=3.5',
    packages=find_packages(),
    install_requires=[
        'plac==0.9.6',
        'numpy==1.14.2',
    ],
    entry_points={
        'console_scripts': [
            'opfy_opf = opfy_split.generate_opf_folds:_entry_point',
            'opfy_numpy = opfy_split.generate_numpy_folds:_entry_point',
        ]
    }
)
