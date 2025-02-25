#!/bin/bash

pip3 install 'torch==2.5.0' 'torchvision==0.20.0' 'torchaudio==2.5.0'
pip3 install packaging 'setuptools>=64' 'setuptools_scm>=8' wheel
pip3 install ninja

pip3 install -e '.[dev,server]' --no-build-isolation --no-use-pep517

pip3 install pre-commit
pre-commit install
pre-commit run --all-files
