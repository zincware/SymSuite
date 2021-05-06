"""
Setup.py file for the SymDet package.
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SymDet",
    version="0.0.1",
    author="Samuel Tovey",
    author_email="tovey.samuel@gmail.com",
    description="Symmetry detection and Lie generator extraction.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SamTov/SymDet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=['numpy',
                      'nbsphinx',
                      'sphinx_rtd_theme',
                      'sphinx-copybutton'],
)