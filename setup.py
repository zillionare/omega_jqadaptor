#!/usr/bin/env python
"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

requirements = [
    "arrow>=0.15",
    "ruamel.yaml>=0.16",
    "numpy>=1.8",
    "jqdatasdk==1.8.9",
    "SQLAlchemy>=1.3.23,<1.4",
    "pytz>=2019.3",
    "cfg4py>=0.8",
    "zillionare-core-types>=0.4.0",
]

setup_requirements = []

test_requirements = []

setup(
    author="Aaron Yang",
    author_email="code@jieyu.ai",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
    ],
    description="Jqdatasdk adapter for zillionare omega",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="zillionare, omega, adaptors, jqdatasdk",
    name="zillionare-omega-adaptors-jq",
    packages=find_packages(include=["jqadaptor", "jqadaptor.*"]),
    setup_requires=setup_requirements,
    url="https://github.com/zillionare/omega_jqadaptor",
    version="1.0.10",
    zip_safe=False,
)
