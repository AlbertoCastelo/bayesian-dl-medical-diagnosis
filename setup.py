"""package entry point"""
from setuptools import setup, find_packages


# with open('requirements.txt') as f:
#     requirements = f.read().splitlines()
#
# with open('requirements-test.txt') as f:
#     test_requirements = f.read().splitlines()

packages = find_packages(include=['deep_gp', 'deep_gp.*'])
setup(
    name='deep_gp',
    version='0.1',
    description='Deep Gaussian Process Classification',
    author='Alberto Castelo',
    packages=packages,
    include_package_data=True,
    # install_requires=requirements
    )
