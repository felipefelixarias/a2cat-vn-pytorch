import os
import goal
import random
import cv2

class GoalImageCache:
    def __init__(self, image_size, dataset_path):
        self.scenes = dict()
        self.cache = dict()
        self.image_size = image_size
        self.random = random.Random()
        pass

    def sample_image(self, collection):
        return self.random.choice([x for x in collection if x.endswith('')])

    def fetch_resource(self, )

    def fetch_scene(self, scene):
        if not scene in self.scenes:
            self.scenes[scene] = sceneobj = dict(
                path = self.fetch_resource('thor-scene-images-%s' % scene),
                resources = dict()
            )

            sceneobj['available_goals'] = os.listdir(sceneobj['path'])

        return self.scenes[scene]

    def all_goals(self, scene):
        return self.fetch_scene(scene)['available_goals']

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
        if (scene, sampled_image) in self.cache:
            return self.cache[(scene, sampled_image)]

        impath = os.path.join(root, sampled_image)
        image = cv2.imread(impath)
        image = cv2.resize(image, self.image_size, interpolation = cv2.INTER_CUBIC)
        self.cache[(scene, sampled_image)] = image
        return image