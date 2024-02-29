from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'eval quality of code'
LONG_DESCRIPTION = 'evaluate quality of code'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="quality_metrics", 
        version=VERSION,
        # author="",
        # author_email="<youremail@email.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Research",
            "Programming Language :: Python :: 3",
            "Operating System :: Ubuntu",
        ]
)
