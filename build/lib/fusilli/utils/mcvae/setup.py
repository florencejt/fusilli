import os
from setuptools import setup, find_packages

src_dir = os.path.join(os.getcwd(), 'src')
packages = {"": "src"}
for package in find_packages("src"):
    packages[package] = "src"

setup(
    packages=packages.keys(),
    package_dir={"": "src"},
    name='mcvae',
    version='2.0.0',
    author='Luigi Antelmi',
    author_email='luigi.antelmi@inria.fr',
    description='Multi-Channel Variational Autoencoder',
    long_description='TODO',
    license='Inria',
    )
