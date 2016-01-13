from setuptools import setup

setup(name='syse',
    version='1.0',
    description='A syntactic sentence extraction program.',
    url='https://github.com/NathanWycoff/SySE',
    author='Nathan Wycoff',
    author_email='nathanbrwycoff@gmail.com',
    packages=['syse'],
    license = 'MIT',
    install_requires = [
        'pandas',
        'numpy',
        'unicodedata',
        'math'],
    dependency_links = ['https://github.com/emilmont/pyStatParser/tarball/master'])
