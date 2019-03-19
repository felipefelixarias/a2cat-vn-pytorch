# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tqdm
import cv2
import numpy as np
import os
import queue
import time
import argparse

from House3D import objrender, Environment, House
from House3D.objrender import RenderMode
from threading import Thread, Lock

RANDOM_SEED = 0

MAX_QSIZE = 20
LOADING_THREADS = 10
RENDER_THREADS = 1

SAMPLES_PER_ROOM = 20
ROOM_TYPES = {'kitchen', 'dining_room', 'living_room', 'bathroom', 'bedroom'}

# Not sure what scale this is, it seems like it's not in meters!
ROBOT_RAD = 0.5
ROBOT_HEIGHT = 1.0

RENDER_MODES = [
    RenderMode.RGB,
    RenderMode.DEPTH,
    RenderMode.SEMANTIC,
    RenderMode.INSTANCE,
    RenderMode.INVDEPTH,
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


def get_house_dir(houseID):
    return os.path.join(output_dir, houseID)


def gen_rand_house(cfg, houses):
    houses = np.array(houses)
    np.random.shuffle(houses)
    for houseID in houses:
        house_dir = get_house_dir(houseID)
        if os.path.exists(house_dir):
            print('{} already exists, skipping'.format(house_dir))
            continue
        yield houseID


def reset_random(env, house, room):
    location = house.getRandomLocationForRoom(room)
    if not location:
        return False

    env.reset(*location)
    return True


def render_current_location(env, houseID, room_type, index):
    output_dir = os.path.join(get_house_dir(houseID), room_type)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print('Created directory {}'.format(output_dir))

    for mode_idx in range(len(RENDER_MODES)):
        render_mode = RENDER_MODES[mode_idx]
        render_name = RENDER_NAMES[mode_idx]

        env.set_render_mode(RENDER_MODES[mode_idx])
        img = env.render_cube_map(copy=True)
        if render_mode == RenderMode.DEPTH:
            img = img[:, :, 0]
        elif render_mode == RenderMode.INVDEPTH:
            img16 = img.astype(np.uint16)
            img = img16[:, :, 0] * 256 + img16[:, :, 1]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        output_filename = 'loc_{}-render_{}.png'.format(index, render_name)
        cv2.imwrite(os.path.join(output_dir, output_filename), img)


def get_valid_rooms(house):
    result = []
    for room in house.all_rooms:
        for tp in room['roomTypes']:
            if tp.lower() in ROOM_TYPES:
                result.append(room)
                break
    return result


def house_loader(house_gen, cfg, house_queue, gen_lock):
    while True:
        while house_queue.qsize() > MAX_QSIZE:
            # Wait until we clear up the queue
            time.sleep(0)

        houseID = None
        with gen_lock:
            try:
                houseID = next(house_gen)
            except StopIteration:
                print('Done processing houses, stopping loading thread...')
                return

        house = None
        try:
            house = create_house(houseID, cfg)
        except Exception as e:
            print('!! Error loading house {}: {}'.format(houseID, e))
            continue

        house_queue.put((houseID, house))
        print('Put house {} in queue, total: {}'.format(houseID,
                                                        house_queue.qsize()))


def house_renderer(cfg, house_queue, progress_queue):
    while True:
        houseID, house = house_queue.get()
        api = objrender.RenderAPIThread(w=512, h=512, device=0)
        env = Environment(api, house, cfg)

        loc_idx = 0
        valid_rooms = get_valid_rooms(house)
        for room in valid_rooms:
            for _i in range(SAMPLES_PER_ROOM):
                if not reset_random(env, house, room):
                    print('Unable to sample location for house {}'.format(
                        houseID))
                    break
                render_current_location(env, houseID, room['id'], loc_idx)
                loc_idx += 1

        house_queue.task_done()
        progress_queue.put(1)
        print('Rendered house {}'.format(houseID))


def progress_tracker(total, progress_queue):
    tracker = tqdm.trange(total)
    while True:
        count = progress_queue.get()
        tracker.update(count)
        progress_queue.task_done()


if __name__ == '__main__':
    output_dir = '/datasets/suncg/render'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), 'houses'), 'r') as f:
        houses = [a.strip() for a in f.readlines()]

    np.random.seed(RANDOM_SEED)

    cfg = {
        "colorFile": '/House3D/House3D/metadata/colormap_coarse.csv',
        "roomTargetFile": '/House3D/House3D/metadata/room_target_object_map.csv',
        "modelCategoryFile": '/House3D/House3D/metadata/ModelCategoryMapping.csv',
        "prefix": '/datasets/suncg/house'
    }


    total = len(houses)
    house_gen = gen_rand_house(cfg, houses)
    gen_lock = Lock()
    house_queue = queue.Queue()
    progress_queue = queue.Queue()

    loader_threads = []
    for _i in range(LOADING_THREADS):
        t = Thread(
            target=house_loader, args=(house_gen, cfg, house_queue, gen_lock))
        t.start()
        loader_threads.append(t)

    render_threads = []
    for _i in range(RENDER_THREADS):
        t = Thread(
            target=house_renderer, args=(cfg, house_queue, progress_queue))
        t.daemon = True
        t.start()
        render_threads.append(t)

    progress_thread = Thread(
        target=progress_tracker, args=(total, progress_queue))
    progress_thread.daemon = True
    progress_thread.start()

    # Wait for queue to be fully populated
    for t in loader_threads:
        t.join()

    # Wait for queue to be fully processed
    house_queue.join()
    print('Done processing!')