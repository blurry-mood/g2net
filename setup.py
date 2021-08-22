from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Read the requirements from the TXT file
with open(path.join(here, 'requirements.txt')) as f:
    requirements = [req for req in f.read().split('\n') if not ('#' in req or req == '')]

setup(
    name='gnet',
    version='0.1.0',
    author='Ayoub Assis',
    author_email='assis.ayoub@gmail.com',
    license='LICENSE',

    packages=find_packages(include=('gnet')),

    install_requires=requirements,
    python_requires='>=3.7',

)