#!/usr/bin/env bash

# edit version in setup.py
rm -f dist/*
python setup.py bdist_wheel
python -m twine upload dist/*
