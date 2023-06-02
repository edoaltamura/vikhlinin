import setuptools
from vikhlinin import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vikhlinin-fit",
    version=__version__,
    description="Fitting routines for cluster profiles in Python.",
    url="https://github.com/edoaltamura/vikhlinin",
    author="Edoardo Altamura",
    author_email="edoardo.altamura@manchester.ac.uk",
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
    scripts=[],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache License Version 2.0 January 2004",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "unyt>=2.9.0"],
    python_requires=">3.8.0",
)
