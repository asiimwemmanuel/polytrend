from setuptools import setup, find_packages

setup(
    name='polytrend',
    version='0.1.0',
    license='GPL-3.0',
    description='Regression algorithm that approximates and plots a polynomial function onto given data',
    long_description='PolyTrend is a regression algorithm that approximates and plots a polynomial function onto given data. It provides insights and conclusions in the fields of interpolation and polynomial regression, specifically in the subfield of approximation theory.',
    author='Emmanuel Asiimwe',
    author_email='asiimwemmanuel47@gmail.com',
    url='https://github.com/asiimwemmanuel/polytrend',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='math, numerical-analysis, data-processing, statistics',
    install_requires=[
        'numpy',
        'scikit-learn',
        'matplotlib',
        'pandas',
        'PyQt6'
    ],
)
