#!/usr/bin/env python

import os
from distutils.core import setup

folder = os.path.dirname(os.path.realpath(__file__))
requirements_path = os.path.join(folder, 'requirements.txt')
install_requires = []
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

setup(name='preprocess_video',
      version='0.1',
      description='Realization of video to frames processing with face detection',
      author='Kishkun Anastasia',
      author_email='',
      package_dir={},
      packages=["preprocess"],
      install_requires=install_requires
      )