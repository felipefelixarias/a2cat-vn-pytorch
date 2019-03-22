#!/usr/bin/python3

from House3D import objrender, Environment, House
from House3D.house import _equal_room_tp
from House3D.objrender import RenderMode
import os
import csv
import cv2
import numpy as np

cfg = {
    "colorFile": os.path.expanduser('~/toolbox/House3D/House3D/metadata/colormap_coarse.csv'),
    "roomTargetFile": os.path.expanduser('~/toolbox/House3D/House3D/metadata/room_target_object_map.csv'),
    "modelCategoryFile": os.path.expanduser('~/toolbox/House3D/House3D/metadata/ModelCategoryMapping.csv'),
    "prefix": os.path.expanduser('/mnt/cluster-home/datasets/suncg/house')
}


ROBOT_RAD = 0.5
ROBOT_HEIGHT = 1.0
ROOM_TYPES = {'kitchen', 'dining_room', 'living_room', 'bathroom', 'bedroom'}
RENDER_MODES = [
    RenderMode.RGB,
    RenderMode.DEPTH,
    RenderMode.SEMANTIC,
    #RenderMode.INSTANCE,
    #RenderMode.INVDEPTH,
]
RENDER_NAMES = ['rgb', 'depth', 'semantic', 'instance', 'invdepth']


class RestrictedHouse(House):
    def __init__(self, **kwargs):
        super(RestrictedHouse, self).__init__(**kwargs)

    def _getRegionsOfInterest(self):
        result = []
        for roomTp in ROOM_TYPES:
            rooms = self._getRooms(roomTp)
            for room in rooms:
                result.append(self._getRoomBounds(room))
        return result



def create_house(houseID, config, robotRadius=ROBOT_RAD):
    print('Loading house {}'.format(houseID))
    objFile = os.path.join(config['prefix'], houseID, 'house.obj')
    jsonFile = os.path.join(config['prefix'], houseID, 'house.json')
    assert (
        os.path.isfile(objFile) and os.path.isfile(jsonFile)
    ), '[Environment] house objects not found! objFile=<{}>'.format(objFile)
    cachefile = os.path.join(config['prefix'], houseID, 'cachedmap1k.pkl')
    if not os.path.isfile(cachefile):
        cachefile = None

    house = RestrictedHouse(
        JsonFile=jsonFile,
        ObjFile=objFile,
        MetaDataFile=config["modelCategoryFile"],
        CachedFile=cachefile,
        RobotRadius=robotRadius,
        SetTarget=False,
        ApproximateMovableMap=True)
    return house

def get_valid_rooms(house):
    result = []
    for room in house.all_rooms:
        for tp in room['roomTypes']:
            x = None
            for y in ROOM_TYPES:
                if _equal_room_tp(tp, y):
                    x = y
                    break
                    
            if x is not None:
                result.append((room, x))
    return result

def load_target_object_data(roomTargetFile):
    room_target_object = dict()
    with open(roomTargetFile) as csvFile:
        reader = csv.DictReader(csvFile)
        for row in reader:
            c = np.array((row['r'],row['g'],row['b']), dtype=np.uint8)
            room = row['target_room']
            if room not in room_target_object:
                room_target_object[room] = []
            room_target_object[room].append(c)
    return room_target_object


def get_valid_room_dict(house):
    valid_rooms = get_valid_rooms(house)
    types_to_rooms = dict()
    for (value, key) in valid_rooms:
        if key in types_to_rooms:
            types_to_rooms[key].append(value)
        else:
            types_to_rooms[key] = [value]
    return types_to_rooms
        
def get_valid_locations(rooms):
    locations = []
    for room in rooms:
        locations.extend([(room, x) for x in house._getValidRoomLocations(room)])
        
    if len(locations) == 0:
        raise Exception('Cannot find the location')
    return locations

def sample_location(house, locations):
    idx = np.random.choice(len(locations))
    return locations[idx][0], house.to_coor(locations[idx][1][0], locations[idx][1][1], True)

def get_target_room_type(room, target):
    for y in room['roomTypes']:
        if _equal_room_tp(y, target):
            return y
        
def is_object_visible(room_target_object, semantic, room_tp):
    resolution = semantic.shape
    total_pixel = resolution[0] * resolution[1]
    n_pixel_for_object_see = max(int(total_pixel * 0.045), 5)
    object_color_list = room_target_object[room_tp]
    _object_cnt = 0
    
    is_visible = False
    for c in object_color_list:
        cur_n = np.sum(np.all(semantic == c, axis=2))
        _object_cnt += cur_n
        if _object_cnt >= n_pixel_for_object_see:
            is_visible = True

    return is_visible, _object_cnt / float(total_pixel)

def sample_true_object(room_target_object, env, house, locations, room_type):
    while True:
        room, location = sample_location(house, locations)
        roomTp = get_target_room_type(room, room_type)

        env.reset(*location)
        semantic = env.render(mode = 'semantic')
        is_visible, _ = is_object_visible(room_target_object, semantic, room_type)
        if is_visible:
            return room, location

def render_current_location(env, houseID, room_type, index, cfg):
    house_dir = os.path.join(os.path.dirname(cfg.get('prefix')),'render2', houseID)
    output_dir = os.path.join(house_dir, room_type)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for mode_idx in range(len(RENDER_MODES)):
        render_mode = RENDER_MODES[mode_idx]
        render_name = RENDER_NAMES[mode_idx]

        env.set_render_mode(RENDER_MODES[mode_idx])
        img = env.render(copy=True)
        if render_mode == RenderMode.DEPTH:
            img = img[:, :, 0]
        elif render_mode == RenderMode.INVDEPTH:
            img16 = img.astype(np.uint16)
            img = img16[:, :, 0] * 256 + img16[:, :, 1]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        output_filename = 'loc_{}-render_{}.png'.format(index, render_name)
        cv2.imwrite(os.path.join(output_dir, output_filename), img)

if __name__ == '__main__':
    from environments.gym_house.env import create_configuration
    from configuration import configuration
    import deep_rl
    from deep_rl.common.console_util import print_progress
    deep_rl.configure(**configuration)
    with open(os.path.join(os.path.dirname(__file__),'jobs', 'houses'), 'r') as f:
        houses = [a.strip() for a in f.readlines()]

    samples_per_room = 20
    screen_size = (512, 512)
    cfg = create_configuration(deep_rl.configuration.get('house3d').as_dict())
    room_target_object = load_target_object_data(cfg['roomTargetFile'])

    api = objrender.RenderAPI(w=screen_size[1], h=screen_size[0], device=0)
    for i, houseID in enumerate(houses):
        print('Processing house %s (%s/%s)' % (houseID, i + 1, len(houses)))
        house = create_house(houseID, cfg)
        env = Environment(api, house, cfg)
        types_to_rooms = get_valid_room_dict(house)
        
        for room_type, rooms in types_to_rooms.items():
            print('Processing house %s (%s/%s) - %s' % (houseID, i + 1, len(houses), room_type))
            
            locations = get_valid_locations(rooms)
            for i in range(samples_per_room):
                #print_progress(0, samples_per_room)
                _, location = sample_true_object(room_target_object, env, house, locations, room_type)
                env.reset(*location)
                render_current_location(env, houseID, room_type, i, cfg)

            #print_progress(samples_per_room, samples_per_room)

        
