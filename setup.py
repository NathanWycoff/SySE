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

    # The project's main homepage.
    url='https://github.com/nathanwycoff/syse',

    # Author details
    author='Nathan Wycoff',
    author_email='nathanbrwycoff@gmail.com',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7'
    ],

    keywords='nlp summarization',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=['syse'],

    install_requires = [
        'pandas',
        'numpy'],


    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        'syse': ['package_data.dat'],
    },
)

