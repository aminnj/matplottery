#!/usr/bin/env bash

# edit version in setup.py
python setup.py bdist_wheel
python -m twine upload dist/*
