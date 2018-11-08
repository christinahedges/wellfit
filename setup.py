#!/usr/bin/env python
import os
import sys
from setuptools import setup

# Prepare and send a new release to PyPI
if "release" in sys.argv[-1]:
    os.system("python setup.py sdist")
    os.system("twine upload dist/*")
    os.system("rm -rf dist/wellfit*")
    sys.exit()


# Load the __version__ variable without importing the package already
exec(open('wellfit/__version__.py').read())

setup(name='wellfit',
      version=__version__,
      description="A wellfit package to fit wells well.",
      long_description=open('README.md').read(),
      author='Christina Hedges',
      author_email='christina.l.hedges@nasa.gov',
      license='MIT',
      package_dir={
            'wellfit': 'wellfit'},
      packages=['wellfit'],
      install_requires=['numpy>=1.11', 'astropy>=1.3', 'scipy>=0.19.0',
                        'matplotlib>=1.5.3', 'tqdm', 'celerite', 'starry',
                        'pandas'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest', 'pytest-cov', 'pytest-remotedata'],
      include_package_data=True,
      classifiers=[
          "Development Status :: 1 - Planning",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Astronomy",
          ],
      )
