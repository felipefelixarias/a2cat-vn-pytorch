import json
import os
import sys
import shutil

if __name__ == "__main__":
    textures = set()

    source_path, source_textures, textures_path = sys.argv[1:]
    os.makedirs(textures_path, exist_ok=True)
    with open(source_path, 'r') as f:
        home = json.load(f)
        for level in home['levels']:
            for node in level['nodes']:
                if not 'materials' in node:
                    continue

                for material in node['materials']:
                    if not 'texture' in material:
                        continue

                    textures.add(material['texture'])
    for texture in textures:
        path = os.path.join(source_textures, '%s.jpg' % texture)
        destination = os.path.join(textures_path, '%s.jpg' % texture)
        if not os.path.exists(destination):
            shutil.copy(path, destination)

    print("%s textures copied" % len(textures))

    