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
        return self.random.choice([x[:-len('-render_rgb.png')] for x in collection if x.endswith('-render_rgb.png')])

    def fetch_scene(self, scene):
        if not scene in self.scenes:
            self.scenes[scene] = sceneobj = dict(
                path = os.path.join(self.dataset_path, scene),
                resources = dict()

            )
            
            sceneobj['available_goals'] = os.listdir(sceneobj['path'])

        return self.scenes[scene]

    def all_goals(self, scene):
        return self.fetch_scene(scene)['available_goals']

    def fetch_image(self, root, scene, resource, sampled_image):
        if (scene, resource, sampled_image) in self.cache:
            return self.cache[(scene, resource, sampled_image)]

        impath = os.path.join(root, sampled_image)
        assert os.path.isfile(impath), ('Missing file %s' % impath)
        image = cv2.imread(impath)
        image = cv2.resize(image, self.image_size, interpolation = cv2.INTER_CUBIC)
        self.cache[(scene, resource, sampled_image)] = image
        return image

    def fetch_resource(self, scene, resource):
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

        return root, images


    def fetch_random(self, scene, resource):
        root, images = self.fetch_resource(scene, resource)       
        sampled_image = self.sample_image(images)        
        return self.fetch_image(root, scene, resource, sampled_image + '-render_rgb.png'), sampled_image

    def fetch_random_with_semantic(self, scene, resource):
        root, images = self.fetch_resource(scene, resource)       
        sampled_image = self.sample_image(images)        
        return (
            self.fetch_image(root, scene, resource, sampled_image + '-render_rgb.png'),
            self.fetch_image(root, scene, resource, sampled_image + '-render_semantic.png')
        ), sampled_image