from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()
      
setup(name='gamma_image',
      version='0.1',
      description='This package contains files to model position and create images of detected gamma rays in a Strontium Iodide crystal sheet.',
      long_description=readme(),
      url='https://github.com/p201-sp2016/gamma_image/',
      author='LaNell Williams and Cody Baxley',
      author_email='lanellwilliams@g.harvard.edu, cbaxley@fas.harvard.edu',
      license=' GNU GPL v3',
      packages=['gamma_image'],
      install_requires=[
          'numpy',
          'pandas',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
