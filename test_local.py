import numpy as np
import jax
import jax.numpy as jnp
import os
import itertools
from timeit import default_timer as timer

import maskgit
from maskgit.utils import visualize_images, read_image_from_url, restore_from_path, draw_image_with_bbox, Bbox
from maskgit.inference import ImageNet_class_conditional_generator

#os.system("mkdir -p checkpoints/")
#
#models_to_download = itertools.product(
#    *[ ["maskgit", "tokenizer"],   [256, 512] ])
#
#for (type_, resolution) in models_to_download:
#  canonical_path = ImageNet_class_conditional_generator.checkpoint_canonical_path(type_, resolution)
#  if os.path.isfile(canonical_path):
#    print(f"Checkpoint for {resolution} {type_} already exists, not downloading again")
#  else:
#    source_url = f'https://storage.googleapis.com/maskgit-public/checkpoints/{type_}_imagenet{resolution}_checkpoint'
#    os.system(f"wget {source_url} -O {canonical_path}")

generator_256 = ImageNet_class_conditional_generator(image_size=256)
generator_512 = ImageNet_class_conditional_generator(image_size=512)
arbitrary_seed = 42
rng = jax.random.PRNGKey(arbitrary_seed)

run_mode = 'normal'  #@param ['normal', 'pmap']

p_generate_256_samples = generator_256.p_generate_samples()
p_edit_512_samples = generator_512.p_edit_samples()

generator_256.maskgit_cf['eval_batch_size'] = 1

path_imgs = "folderOnColab/coco_sq/images_resize256_crop256"
path_labels = "folderOnColab/coco_sq/imagenet_resize256_crop256"
path_bbox = "folderOnColab/coco_sq/bbox_resize256_crop256"
path_out = "folderOnColab/coco_sq/maskgit_results"
if not os.path.exists(path_out):
  os.mkdir(path_out)
filenames = os.listdir("folderOnColab/coco_sq/images_resize256_crop256")
#label = int(category.split(')')[0])
# filename = filenames[0]
# we switch to 512 here for demo purposes
image_size = 256
