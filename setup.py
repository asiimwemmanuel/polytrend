import sys
from setuptools import setup, find_packages
from cx_Freeze import Executable

# Project Metadata
setup(
	name='PolyTrend',
	version='0.1.0',
	license='GPL-3.0',
	description='Regression algorithm for polynomial function approximation',
	long_description='PolyTrend is a regression algorithm that approximates and plots a polynomial function onto given data. It provides insights and conclusions in the fields of interpolation and polynomial regression, specifically in the subfield of approximation theory.',
	author='Emmanuel Asiimwe',
	author_email='asiimwemmanuel47@gmail.com',
	url='https://github.com/asiimwemmanuel/polytrend',
	packages=find_packages(),
	classifiers=[
		'Development Status :: 4 - Beta',
		'Intended Audience :: Developers',
		'License :: OSI Approved :: GNU General Public License v3 (GPL-3)',
		'Programming Language :: Python :: 3.6',
		'Programming Language :: Python :: 3.11',
		'Operating System :: Microsoft :: Windows',
		'Operating System :: POSIX :: Linux',
		# 'Operating System :: Apple :: macOS'
	],
	keywords='math, numerical-analysis, data-processing, statistics',
	install_requires=[
		'numpy',
		'scikit-learn',
		'matplotlib',
		'pandas',
		'PyQt6',
		'PySide6'
	],
	options = {
		'build_exe': {
			'packages': ['numpy', 'sklearn', 'matplotlib', 'pandas', 'PyQt6', 'PySide6'],
			'includes': ['os', 'random', 'datetime', 'typing', 'sys'],
			'excludes': ['Qt_6.5'],
			'add_data': [
				('./src/polytrend.py', 'src'),
				('./gui/gui.py', 'gui'),
			],
		}
	}
)

# Executable Definition
executables = [
	Executable(
		'main.py',
		base='Win32GUI' if sys.platform == 'win32' else None,
	)
]
