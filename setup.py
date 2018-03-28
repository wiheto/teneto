"""
General setup for module
"""

from setuptools import setup, find_packages

setup(name='Teneto',
      version='0.2.8',
      python_requires='>3.5',
      install_requires=[  
			'nilearn>=0.4.0',
			'statsmodels>=0.8.0',
			'pybids>=0.4.2'],
      description='Temporal network tools',
      packages=find_packages(),
      author='William Hedley Thompson',
      author_email='hedley@startmail.com',
      url='https://www.github.com/wiheto/teneto',
      download_url='https://github.com/wiheto/teneto/archive/0.2.tar.gz',
      package_data={'':['./teneto/data']},
      include_package_data = True)
