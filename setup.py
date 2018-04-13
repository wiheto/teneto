"""
General setup for module
"""

from setuptools import setup, find_packages

setup(name='teneto',
      version='0.3.0',
      python_requires='>3.5',
      install_requires=[  
			'nilearn>=0.4.0',
			'statsmodels>=0.8.0',
			'pybids>=0.4.2',
      'louvain>=0.6.1',
      'python-igraph>=0.6.1'],
      description='Temporal network tools',
      packages=find_packages(),
      author='William Hedley Thompson',
      author_email='hedley@startmail.com',
      url='https://www.github.com/wiheto/teneto',
      download_url='https://github.com/wiheto/teneto/archive/0.2.tar.gz',
      package_data={'':['./teneto/data']},
      include_package_data = True)
