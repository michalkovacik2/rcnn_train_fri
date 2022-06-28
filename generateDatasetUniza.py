# This code is based on: https://github.com/chriskhanhtran/object-detection-detectron2
# This code only creates train dataset for UNIZA dataset.
# On OpenImages dataset, the code was used to create train and validation datasets. That code is basically the same
# as in the github. Only need to change the Prepare annotation files (1), part to include validation annotations.
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
from tqdm import tqdm
import os, cv2, random
import numpy as np
import pandas as pd

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

# All classes for training
target_classes = ['Bus', 'Car', 'Truck', 'Van', 'Inside', 'CloseUp']
class2id = {k: v for v, k in enumerate(target_classes)}
print(class2id)

# Get LabelName of target classes
classes = pd.read_csv("annotations/class-descriptions-boxable.csv", header=None, names=['LabelName', 'Class'])
subset_classes = classes[classes['Class'].isin(target_classes)]
print(subset_classes)

file_name_corrected_results_uniza = "corrected_result_uniza-annotations-bbox-truncated.csv"
file_name_annotations = "/home/mickov/Documents/PROJEKT1/DATASET_TRAIN_AGAIN2/annotations/corrected_result_uniza.csv"

# Prepare annotation files (1)
for folder in ['train']:
    # Load data
    # This path needs to be changed in order to work
    # This csv file needs to use OpenImage format
    annot_df = pd.read_csv(file_name_annotations)
    # Inner join to keep only target classes
    annot_df = annot_df.merge(subset_classes, on='LabelName')
    # Create `ClassID` column
    annot_df['ClassID'] = annot_df['Class'].apply(lambda x: class2id[x])
    # Save truncated annot_df
    annot_df.to_csv(file_name_corrected_results_uniza, index=False)
    del annot_df

'''
# For train and validation dataset
# Prepare annotation files
for folder in ['train', 'validation']:
    # Load data
    annot_df = pd.read_csv(f"{folder}-annotations-bbox.csv")
    # Inner join to keep only target classes
    annot_df = annot_df.merge(subset_classes, on='LabelName')
    # Create `ClassID` column
    annot_df['ClassID'] = annot_df['Class'].apply(lambda x: class2id[x])
    # Save truncated annot_df
    annot_df.to_csv(f"{folder}-annotations-bbox-truncated.csv", index=False)
    del annot_df
'''
    
def denormalize_bboxes(bboxes, height, width):
    """Denormalize bounding boxes in format of (xmin, ymin, xmax, ymax)."""
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * width
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * height
    return np.round(bboxes)


def get_detectron_dicts(annot_df):
    """
    Create Detectron2's standard dataset from an annotation file.
    
    Args:
        annot_df (pd.DataFrame): annotation dataframe.
    Return:
        dataset_dicts (list[dict]): List of annotation dictionaries for Detectron2.
    """
    
    # Get image ids
    img_ids = annot_df["ImageID"].unique().tolist()
    
    dataset_dicts = []
    for img_id in tqdm(img_ids):
        # PATH to the images
        file_name = f'images/{img_id}.jpg'
        if not os.path.exists(file_name):
            print(f'File {file_name} does not exist.')
            continue
            
        height, width = cv2.imread(file_name).shape[:2]
            
        record = {}
        record['file_name'] = file_name
        record['image_id'] = img_id
        record['height'] = height
        record['width'] = width
        
        # Extract bboxes from annotation file
        bboxes = annot_df[['XMin', 'YMin', 'XMax', 'YMax']][annot_df['ImageID'] == img_id].values
        bboxes = denormalize_bboxes(bboxes, height, width)
        class_ids = annot_df[['ClassID']][annot_df['ImageID'] == img_id].values
        
        annots = []
        for i, bbox in enumerate(bboxes.tolist()):
            annot = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": int(class_ids[i]),
            }
            annots.append(annot)

        record["annotations"] = annots
        dataset_dicts.append(record)
    return dataset_dicts


# Load subset annotations
# This is the same string as above
train_df = pd.read_csv(file_name_corrected_results_uniza)

data_size = pd.concat([train_df['Class'].value_counts()], axis=1)
data_size.columns = ["Train"]
print(data_size)

# Register dataset with Detectron2
DatasetCatalog.register("car_train", lambda d: get_detectron_dicts(train_df))
MetadataCatalog.get("car_train").set(thing_classes=target_classes)

# No validation set was created here.
# DatasetCatalog.register("car_val", lambda d: get_detectron_dicts(val_df))
# MetadataCatalog.get("car_val").set(thing_classes=target_classes)

# Get metadata. It helps show class labels when we visualize bounding boxes
car_train_metadata = MetadataCatalog.get("car_train")

dataset_dicts = get_detectron_dicts(train_df)

# Shows a random sample from the created dataset to identify if successfull
for d in random.sample(dataset_dicts, 20):
    img = cv2.imread(d["file_name"])
    print(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=car_train_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow('img', out.get_image()[:, :, ::-1])
    cv2.waitKey(0)

'''
# This code here was used for training on the uniza dataset, in which the images were in a lot of subfolders.
# So copied every image to one folder and renamed them nameOfSubfolder_imageName.jpg == 280465965_pSMAAOSw7PNd5aWF.jpg
# is image in folder 280465965, this code then removes the folder name for the images that have them 280465965/ 

file = open("corrected_result_uniza-annotations-bbox-truncated.csv", 'r')
outfile = open('transformed_annot.csv', 'w')

ind = 0
for line in file:
    if ind == 0:
        outfile.write(line)
    else:
        imageId = line.split(',')[0]
        splitted = line.split(',')[1:]
        if len(imageId.split('/')) == 1:
            imageIdNew = imageId.split('/')[0]
        else:
            imageIdNew = imageId.split('/')[1]
        string = ','.join(splitted)
        string = imageIdNew + ',' + string
        outfile.write(string)

    ind += 1
'''
