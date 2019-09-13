#!/usr/bin/env python
from setuptools import setup
import glob
from acis_thermal_ml import __version__

templates = glob.glob("templates/*")
scripts = glob.glob("scripts/*")

url = 'https://github.com/acisops/acis_thermal_ml/tarball/{}'.format(__version__)

setup(name='acis_thermal_ml',
      packages=["acis_thermal_ml"],
      version=__version__,
      description='ACIS Thermal Model Machine Learning Package',
      author='John ZuHone',
      author_email='john.zuhone@cfa.harvard.edu',
      url='http://github.com/acisops/acis_thermal_ml',
      download_url=url,
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.6',
      ],
      )