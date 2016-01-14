from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='syse',

    version='1.1.1',

    description='A syntactic sentence extraction program.',
    long_description=long_description,

    url='https://github.com/nathanwycoff/syse',

    author='Nathan Wycoff',
    author_email='nathanbrwycoff@gmail.com',

    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7'
    ],

    keywords='nlp summarization',

    packages=['syse'],

    install_requires = [
        'pandas',
        'numpy'],
    package_data={
        'syse': ['default.dat'],
    },
)

