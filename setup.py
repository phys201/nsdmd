from setuptools import setup

setup(name='nsdmd',
      version='1',
      description='Tool for calculating rotational curve by DarkMatterDesensityFiles',
      url='https://github.com/p201-sp2016/NestedSampling_DMD',
      author='Daniya and Jun',
      author_email='yinjun02@gmail.com and daniya_seitova@g.harvard.edu ',
      license='GNU General Public License v3.0',
      packages=['nsdmd'],
      install_requires=[
          'numpy',
          'nestle',
          'corner',
          'matplotlib',
          'scipy',
          'pandas' 
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False,
)
