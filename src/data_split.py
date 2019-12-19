import os, shutil

original_abdomen_dir = './Data/Augmented/Abdomen'
original_femur_dir = './Data/Augmented/Femur'
original_head_dir = './Data/Augmented/Head'

base_dir = './Data/Split'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_abdomen_dir = os.path.join(train_dir, 'Abdomen')
os.mkdir(train_abdomen_dir)

train_femur_dir = os.path.join(train_dir, 'Femur')
os.mkdir(train_femur_dir)

train_head_dir = os.path.join(train_dir, 'Head')
os.mkdir(train_head_dir)

validation_abdomen_dir = os.path.join(validation_dir, 'Abdomen')
os.mkdir(validation_abdomen_dir)

validation_femur_dir = os.path.join(validation_dir, 'Femur')
os.mkdir(validation_femur_dir)

validation_head_dir = os.path.join(validation_dir, 'Head')
os.mkdir(validation_head_dir)

test_abdomen_dir = os.path.join(test_dir, 'Abdomen')
os.mkdir(test_abdomen_dir)

test_femur_dir = os.path.join(test_dir, 'Femur')
os.mkdir(test_femur_dir)

test_head_dir = os.path.join(test_dir, 'Head')
os.mkdir(test_head_dir)

fnames = ['Abdomen_{:05d}.jpg'.format(i) for i in range(600)]
for fname in fnames:
    src = os.path.join(original_abdomen_dir, fname)
    dst = os.path.join(train_abdomen_dir, fname)
    shutil.copy(src, dst)

fnames = ['Abdomen_{:05d}.jpg'.format(i) for i in range(600, 800)]
for fname in fnames:
    src = os.path.join(original_abdomen_dir, fname)
    dst = os.path.join(validation_abdomen_dir, fname)
    shutil.copy(src, dst)
    
fnames = ['Abdomen_{:05d}.jpg'.format(i) for i in range(800, 1000)]
for fname in fnames:
    src = os.path.join(original_abdomen_dir, fname)
    dst = os.path.join(test_abdomen_dir, fname)
    shutil.copy(src, dst)

fnames = ['Femur_{:05d}.jpg'.format(i) for i in range(600)]
for fname in fnames:
    src = os.path.join(original_femur_dir, fname)
    dst = os.path.join(train_femur_dir, fname)
    shutil.copy(src, dst)

fnames = ['Femur_{:05d}.jpg'.format(i) for i in range(600, 800)]
for fname in fnames:
    src = os.path.join(original_femur_dir, fname)
    dst = os.path.join(validation_femur_dir, fname)
    shutil.copy(src, dst)
    
fnames = ['Femur_{:05d}.jpg'.format(i) for i in range(800, 1000)]
for fname in fnames:
    src = os.path.join(original_femur_dir, fname)
    dst = os.path.join(test_femur_dir, fname)
    shutil.copy(src, dst)

fnames = ['Head_{:05d}.jpg'.format(i) for i in range(600)]
for fname in fnames:
    src = os.path.join(original_head_dir, fname)
    dst = os.path.join(train_head_dir, fname)
    shutil.copy(src, dst)

fnames = ['Head_{:05d}.jpg'.format(i) for i in range(600, 800)]
for fname in fnames:
    src = os.path.join(original_head_dir, fname)
    dst = os.path.join(validation_head_dir, fname)
    shutil.copy(src, dst)
    
fnames = ['Head_{:05d}.jpg'.format(i) for i in range(800, 1000)]
for fname in fnames:
    src = os.path.join(original_head_dir, fname)
    dst = os.path.join(test_head_dir, fname)
    shutil.copy(src, dst)