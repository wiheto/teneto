"""
General setup for module
"""

from setuptools import setup, find_packages

setup(name='Teneto',
      version='0.2.7b',
      install_requires=['setuptools>=38.5.2',
			'numpy>=1.14.1',
			'nilearn>=0.4.0',
			'statsmodels>=0.8.0',
			'seaborn>=0.8.1',
			'matplotlib>=2.2.0',
			'scipy>=1.0.0',
			'pandas>=0.22',
			'scikit-learn>=0.19.1',
			'pybids>=0.4.2'],
      description='Temporal network tools',
      packages=find_packages(),
      author='William Hedley Thompson',
      author_email='hedley@startmail.com',
      url='https://www.github.com/wiheto/teneto',
      download_url='https://github.com/wiheto/teneto/archive/0.2.tar.gz',
      package_data={'':['./teneto/data']},
      include_package_data = True)
