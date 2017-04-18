from setuptools import setup

setup(name='NestedSampling_DMD',
      version='1',
      description='Tool for calculating rotational curve by DarkMatterDesensityFiles',
      url='https://github.com/p201-sp2016/NestedSampling_DMD',
      author='Daniya and Jun',
      author_email='yinjun02@gmail.com', 'daniya_seitova@g.harvard.edu ',
      license='GNU General Public License v3.0',
      packages=['NestedSampling_DMD'],
      install_requires=[
          'numpy',
          'nestle',
          'math',
          'corner',
          'matplotlib',
          'scipy'
            
      ],
      zip_safe=False,
)
