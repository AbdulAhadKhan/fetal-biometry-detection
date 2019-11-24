import os
from PIL import Image, ImageOps

source_dir = './Data/Original'
dest_dir = './Data/Converted'
if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

classes = os.listdir(source_dir)
source_classes_dir = [os.path.join(source_dir, d) for d in classes]
dest_classes_dir = [os.path.join(dest_dir, d) for d in classes]

for directory in dest_classes_dir:
    if not os.path.exists(directory):
        os.mkdir(directory)

for i in range(0, len(classes)):
    names = os.listdir(source_classes_dir[i])
    for name in names:
        src = os.path.join(source_classes_dir[i], name)
        name = name.split('.')[0] + '.jpg'
        dst = os.path.join(dest_classes_dir[i], name)
        image = Image.open(src).convert('L')
        image = ImageOps.fit(image, (200, 200))
        image.save(dst)