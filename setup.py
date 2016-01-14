from setuptools import setup, find_packages

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
        'numpy'],
    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '/home/johnmadden/syse/': ['default','README','LICENSE']
    },
    include_package_data = True,
    dependency_links = ['https://github.com/emilmont/pyStatParser/tarball/master'])
