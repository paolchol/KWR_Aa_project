import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

INSTALL_REQUIRES = [
      'numpy',
      'pandas'
]

setuptools.setup(
    name="PCTimeseriesLSTMAA",          # This is the name of the package
    version="0.9",                        # The initial release version
    author="PColombo",                     # Full name of the author
    author_email='paolocolombo1996@gmail.com',
    description="TS analysis on Aa",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    # packages=setuptools.find_packages(exclude=['tests*']),    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
    packages=['PCTimeseriesLSTMAA', 'PCTimeseriesLSTMAA.main'],
    # py_modules=['main'],
    install_requires=INSTALL_REQUIRES,                   # Install other dependencies if any
    include_package_data=True,

    # url
)
