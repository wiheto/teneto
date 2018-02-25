"""
Gemeral setup for module
"""

from setuptools import setup, find_packages

setup(name='Teneto',
      version='0.2.5',
      description='Temporal network tools',
      packages=find_packages(),
      author='William Hedley Thompson',
      author_email='hedley@startmail.com',
      url='https://www.github.com/wiheto/teneto',
      download_url='https://github.com/wiheto/teneto/archive/0.2.tar.gz',
      package_data={'':['./teneto/data']},
      include_package_data = True)
