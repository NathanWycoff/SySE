from setuptools import setup, find_packages

setup(name='syse',
    version='1.1',
    description='A syntactic sentence extraction program.',
    url='https://github.com/NathanWycoff/SySE',
    author='Nathan Wycoff',
    author_email='nathanbrwycoff@gmail.com',
    packages=['syse'],
    license = 'MIT',
    install_requires = [
        'pandas',
        'numpy'],
    package_data={
        'syse': ['default'],
    }
    include_package_data = True,
    dependency_links = ['https://github.com/emilmont/pyStatParser/tarball/master'])
