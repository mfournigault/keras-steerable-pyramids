import sys
from setuptools import setup, find_packages

setup(
    name='keras_steerable_pyramids',
    version='0.2',
    author='Tom Runia, Modified for Keras',
    author_email='tomrunia@gmail.com',
    url='https://github.com/mfournigault/keras-steerable-pyramids',
    description='Complex Steerable Pyramids in Keras 3 + TensorFlow 2',
    long_description='Fast CPU/CUDA implementation of the Complex Steerable Pyramid in Keras 3 with TensorFlow 2 backend.',
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'tensorflow>=2.15.0',
        'keras>=3.0.0',
        'matplotlib',
        'pillow',
        'scikit-image',
    ],
    python_requires='>=3.10',
    scripts=[]
)