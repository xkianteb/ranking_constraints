from setuptools import setup, find_packages

with open("requirements.txt") as fp:
    requirements = fp.read().splitlines()

setup(
    name="constraint_ranking",
    version="0.1.0",
    packages=find_packages(),
    authors="",
    python_requires=">=3.10",
    install_requires=requirements,
)
