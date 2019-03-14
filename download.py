import requests
import io
import zipfile
from collections import namedtuple
import os

DownloaderContext = namedtuple('DownloaderContext', ['base_url', 'resources_path'])

class Downloader:
    def __init__(self):
        self.base_url = 'https://deep-rl.herokuapp.com/resources/'
        self.resources = dict()
        self._base_path = None

    @property
    def base_path(self):
        if self._base_path is None:
            self._base_path = os.path.expanduser('~/.visual_navigation')
        return self._base_path

    @property
    def resources_path(self):
        return os.path.join(self.base_path, 'resources')

    def create_context(self):
        return DownloaderContext(self.base_url, self.resources_path)

    def add_resource(self, name, fn):
        self.resources[name] = fn

    def get(self, name):
        return self.resources[name](self.create_context())

downloader = Downloader()

def download_resource(name, context):
    resource_path = os.path.join(context.resources_path, name)
    if os.path.exists(resource_path):
        return resource_path

    url = context.base_url + '%s.zip' % name
    try:
        print('Downloading resource %s.' % name)  
        response = requests.get(url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(resource_path)

        print('Resource %s downloaded.' %name)
        return resource_path

    except Exception as e:
        if os.path.exists(resource_path):
            os.remove(resource_path)
        raise e

def download_resource_task(name):
    def thunk(context):
        return download_resource(name, context)
    return thunk

SCENES_NUMER = 340
for scene_number in range(SCENES_NUMER):
    resource = 'thor-scene-images-%s' % scene_number
    downloader.add_resource(resource, download_resource_task(resource))
downloader.add_resource('test', download_resource_task('test'))

def get_resource(name):
    return downloader.get(name)