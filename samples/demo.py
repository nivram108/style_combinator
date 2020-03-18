import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import fast
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco
src_path = "src/"
output_path = "output.png"
# from mrcnn import utils
model = None
file_names = ""
images = None
results = None
r = None
class_names = []
class_style_dict = {"background" : "none"}
style_options = ["adventure_time", "ff", "food_cube", "sun_set", "the_scream", "candy", "oil", "wave", "wind_valley", "none"]
contained_classes = []
collapse_masks = None
collapse_id = None
def setup():
    global model
    global class_names
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Directory of images to run detection on
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")

    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    # config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']
    print("done setup")


def set_input_image(f):
    global file_names
    global image
    global results
    global r
    file_names = f
    image = skimage.io.imread(file_names)
    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]

def combine(mask0, mask1):
    for y in range(0, len(mask0)):
        for x in range(0, len(mask0[0])):
            mask0[y, x] = mask0[y, x] or mask1[y, x]
def set_output_path(path):
    global output_path
    output_path = path
def get_file_name(file_name, style):
    file_name = file_name.split(".")[0]
    if style == "result":
        return output_path
    return src_path + file_name + "_" + style + ".png"
def draw_mask(file_name, image, mask):
    origin = Image.open(file_name)
    for y in range(0, len(mask)):
        for x in range(0, len(mask[0])):
            if mask[y,x] == True:
                image.putpixel((x,y), origin.getpixel((x,y)))
def generate_style(original_file_name, class_names):
    file_name = original_file_name.split(".")[0]
    # class_names.append("background")
    for class_name in class_names:
        style_name = get_style_name(class_name)
        if not style_name == "none":
            fast.generate_image(original_file_name, style_name, get_file_name(file_name, style_name))
    # fast.generate_image(original_file_name, get_style_name("background"), get_file_name(file_name, get_style_name("background")))

def get_style_name(class_name):
    global class_style_dict
    if class_name in class_style_dict.keys():
        return class_style_dict[class_name]
    else:
        return "none"

def show_contained_classes():
    global contained_classes
    print("contain class:")
    for class_name in contained_classes:
        print(class_name)
def show_style_options():
    global style_options
    print("styles:")
    for style in style_options:
        print(style)
def set_style(styles):
    global class_style_dict
    # for key in styles.keys():
    #     class_style_dict[key] = styles[key]
    class_style_dict = styles

def classify():
    global r
    global file_names
    global contained_classes
    global collapse_masks
    global collapse_id
    collapse_masks = []
    collapse_masks.append(r['masks'][:, :, 0])
    collapse_id = {r['class_ids'][0]: 0}
    # styles = {"background":"oil", "person" : "food", "bench":"wave", "handbag" : "candy"}
    collapse_class_names = [class_names[r['class_ids'][0]]]
    for i in range(0, len(r['class_ids'])):
        class_id = r['class_ids'][i]
        if class_id in collapse_id.keys():
            combine(collapse_masks[collapse_id[class_id]], r['masks'][:, :, i])
        else:
            collapse_id[class_id] = len(collapse_masks)
            collapse_masks.append(r['masks'][:, :, i])
    for id in r['class_ids']:
        print("class id:" + str(class_names[id]))
        if not class_names[id] in collapse_class_names:
            collapse_class_names.append(class_names[id])
    contained_classes = collapse_class_names
    contained_classes.append("background")

def generate():
    global contained_classes
    global file_names
    global class_names
    global collapse_masks
    global collapse_id
    generate_style(file_names, contained_classes)
    Image.open(file_names).save(get_file_name(file_names, "none"))
    background = Image.open(get_file_name(file_names, get_style_name("background")))
    background = background.convert("RGBA")
    for class_id in collapse_id.keys():
        print("class id :" + str(class_id))
    #     im = Image.new("RGBA", (len(mask0[0]), len(mask0)))
        draw_mask(get_file_name(file_names, get_style_name(class_names[class_id])), background, collapse_masks[collapse_id[class_id]])
    #     im.save(get_file_name(file_names, class_names[class_id]))
    #     print(im.mode)
    #     print(background.mode)
    #     background = Image.blend(background, im, 1)

    background.save(get_file_name(file_names, "result"))
    # draw_mask(im, mask0)

    #         line = line + str(int(mask0[x, y]))
    # im.save('simplePixel.png') # or any image format
    # print("r rois:" + str(len(r['rois'])))
    # print("r scores:" + str(len(r['scores'])))
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
    #                             class_names, r['scores'])
def get_style_options():
    global style_options
    return style_options
def get_contained_class():
    global contained_classes
    return contained_classes

def main():
    setup()
    set_input_image(input("input file name:"))
    classify()
    show_style_options()
    show_contained_classes()
    set_style()
    generate()
