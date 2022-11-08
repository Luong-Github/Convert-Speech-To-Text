from distutils.command.install_scripts import install_scripts
import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="Speech2T",
    py_modules=["Speech2T"],
    description="Processing NLP problem from sound to text",
    author="luong tran",
    author_email="luong.locery@gmail.com",
    python_requires='>3.7',
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), 'requirement.txt'))
        )
    ]
)

os.chmod()