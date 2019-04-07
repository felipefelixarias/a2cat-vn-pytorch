from download import downloader
from importlib import import_module
import os

if __name__ == '__main__':
    for package in os.listdir('experiments'):
        if not package.endswith('.py'):
            continue
        package = package[:-3]
        import_module('experiments.' + package)
    import_module('environments')

    downloader.download_all()