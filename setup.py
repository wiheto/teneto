"""General setup for module."""

from setuptools import setup, find_packages

VERSION = "teneto/_version.py"
VERSION = open(VERSION, "rt").read()
VERSION = VERSION.split('"')[1]

setup(name='teneto',
      version=VERSION,
      python_requires='>3.5',
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      install_requires=[
          ## Required
          'statsmodels>=0.8.0',
          'networkx>=2.0',
          'python-louvain>=0.13',
          'pandas>=0.21',
          'scipy>=1.4.1',
          'numpy>=1.16.1',
          'matplotlib >=3.1',
          'tables>=3.4.4'
          ## Optional for neural
          'scikit-learn>=0.20.3'
          'nilearn>=0.6.0',
          'pybids>=0.9',
          'templateflow>=0.4.1',
          ],
      description='Temporal network tools',
      packages=find_packages(),
      author='William Hedley Thompson',
      author_email='hedley@startmail.com',
      url='https://www.github.com/wiheto/teneto',
      download_url='https://github.com/wiheto/teneto/archive/0.3.3.tar.gz',
      package_data={'': ['./teneto/data']},
      include_package_data=True,
      entry_points={
          'console_scripts': ['teneto = teneto.__main__:main']
      },
      long_description='Temporal network tools. \
        A package for deriving, analysing and plotting temporal network representations. \
        Additional tools for temporal network analysis with neuroimaging contexts.')
