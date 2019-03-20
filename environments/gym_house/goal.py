import os
import random
import cv2

class GoalImageCache:
    def __init__(self, image_size, dataset_path):
        self.scenes = dict()
        self.cache = dict()
        self.image_size = image_size
        self.dataset_path = os.path.join(dataset_path, 'render')
        self.random = random.Random()
        pass

    def sample_image(self, collection):
        return self.random.choice([x for x in collection if x.endswith('-render_rgb.png')])

    def fetch_scene(self, scene):
        if not scene in self.scenes:
            self.scenes[scene] = sceneobj = dict(
                path = os.path.join(self.dataset_path, scene),
                resources = dict()
            )

        return self.scenes[scene]

    def fetch_random(self, scene, resource):
        self.fetch_scene(scene)

        if not resource in self.scenes[scene]['resources']:
            root = os.path.join(self.scenes[scene]['path'], resource)
            images = os.listdir(root)
            self.scenes[scene]['resources'][resource] = dict(
                root = root,
                images = images
            )
        else:
            row = self.scenes[scene]['resources'][resource]
            root, images = row['root'], row['images']

        sampled_image = self.sample_image(images)
        if (scene, resource, sampled_image) in self.cache:
            return self.cache[(scene, resource, sampled_image)]

        impath = os.path.join(root, sampled_image)
        assert os.path.isfile(impath), ('Missing file %s' % impath)
        image = cv2.imread(impath)
        image = cv2.resize(image, self.image_size, interpolation = cv2.INTER_CUBIC)
        self.cache[(scene, resource, sampled_image)] = image
        return image