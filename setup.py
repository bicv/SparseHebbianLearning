from setuptools import setup, find_packages

setup(name='SHL_scripts',
      version='0.0.0.0',
      author='',
      description='SHL_scripts description',
      long_description=open('README.rst').read(),
      license='LICENSE.txt',
      keywords="",

      # package source directory
      package_dir={'': 'src'},
      packages=find_packages('src', exclude='docs')


)