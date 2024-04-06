# https://stackoverflow.com/a/62238104/17521658
# answered by Jacob (Jun 6, 2020 at 21:13)

from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_data_files
hiddenimports = collect_submodules('scipy')

datas = collect_data_files('scipy')