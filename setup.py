from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='matplottery',
    version='1.0.3',
    description='Nicer histograms with numpy and plotting with matplotlib',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/aminnj/matplottery',
    author='Nick Amin',
    author_email='amin.nj@gmail.com',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
    ],
    packages = find_packages(exclude = ["tests","examples"]),
    test_suite="tests",
    install_requires=[
        "matplotlib>2.0",
        "numpy",
        ],
)
