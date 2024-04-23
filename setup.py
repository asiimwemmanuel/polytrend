import sys
from setuptools import setup, find_packages
from cx_Freeze import Executable

# Project Metadata
setup(
	name='PolyTrend',
	version='1.0.0',
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
		'Programming Language :: Python :: 3.11',
		'Operating System :: Microsoft :: Windows',
		'Operating System :: POSIX :: Linux',
		'Operating System :: Apple :: macOS'
	],
	keywords='math, numerical-analysis, data-processing, statistics',
	install_requires=[
		'cx_Freeze==6.15.16',
		'matplotlib==3.8.3',
		'numpy==1.26.4',
		'scipy==1.12.0',
		'pandas==2.2.1',
		'PyQt6==6.6.1',
		'PyQt6_sip==13.6.0',
		'scikit_learn==1.4.1.post1',
		'setuptools==69.5.1'
	],
	options = {
		'build_exe': {
			'packages': ['numpy', 'sklearn', 'matplotlib', 'pandas', 'PyQt6', 'PySide6'],
			'includes': ['os', 'random', 'datetime', 'typing', 'sys'],
			'excludes': [
                'Qt_6.5',
                'whichcraft', 
                'bottle', 
				'altgraph', 
				'zope.interface', 
				'zope.event', 
				'urllib3', 
				'pywin32-ctypes', 
				'pyinstaller-hooks-contrib', 
				'pycparser', 
				'pefile', 
				'idna', 
				'greenlet', 
				'future', 
				'charset-normalizer', 
				'certifi', 
				'requests', 
				'pyinstaller', 
				'cffi', 
				'gevent', 
				'gevent-websocket', 
				'bottle-websocket', 
				'Eel', 
				'auto-py-to-exe'
            ],
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
