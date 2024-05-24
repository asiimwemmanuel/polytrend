from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

# Project Metadata
setup(
    name='PolyTrend',
    version='1.1.0',
    license='GPL-3.0',
    description='Regression algorithm for polynomial function approximation',
    long_description='PolyTrend is a Python app aimed at facilitating polynomial trend fitting, visualization, and extrapolation. It offers a comprehensive set of functionalities to analyze and interpret data using polynomial regression techniques. Its development provides insights and conclusions in the fields of interpolation, polynomial regression and approximation theory.',
    author='Emmanuel Asiimwe',
    author_email='asiimwemmanuel47@gmail.com',
    url='https://github.com/asiimwemmanuel/polytrend',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPL-3)',
        'Programming Language :: Python :: 3',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Apple :: macOS'
    ],
    keywords='math, numerical-analysis, data-processing, statistics',
    install_requires=[
        'cx_Freeze>=6.15.16',
        'matplotlib>=3.0.0',
        'numpy>=1.16.0',
        'scipy>=1.2.0',
        'PyQt5>=5.15.10',
        'scikit-learn>=0.24.0',
        'setuptools>=30.3.0'
    ],
    ext_modules=cythonize([
        Extension("src.polytrend", ["src/polytrend.py"]),
        Extension("src.view.gui_ui", ["src/view/gui_ui.py"]),
    ],
    build_dir="build_cythonize",
    compiler_directives={
        'language_level': "3",
        'always_allow_keywords': True,
    }),
    cmdclass={'build_ext': build_ext},
    options={
        'build_exe': {
            'packages': ['numpy', 'sklearn', 'matplotlib', 'PyQt5'],  # or 'PySide6'
            'includes': ['os', 'random', 'datetime', 'typing', 'sys'],
            'excludes': [
                'Qt_6.5', 'whichcraft', 'bottle', 'altgraph', 'zope.interface',
                'zope.event', 'urllib3', 'pywin32-ctypes', 'pyinstaller-hooks-contrib',
                'pycparser', 'pefile', 'idna', 'greenlet', 'future', 'charset-normalizer',
                'certifi', 'requests', 'pyinstaller', 'cffi', 'gevent', 'gevent-websocket',
                'bottle-websocket', 'Eel', 'auto-py-to-exe'
            ],
            'include_files': [ # Corrected from 'add_data' to 'include_files'
                ('./src/polytrend.py', 'src'),
                ('./gui/gui.py', 'gui'),
            ],
        }
    },
    python_requires='>=3.11'
)

# Uncomment and configure if creating executables using cx_Freeze
# executables = [
#     Executable(
#         'main.py',
#         base='Win32GUI' if sys.platform == 'win32' else None,
#     )
# ]
