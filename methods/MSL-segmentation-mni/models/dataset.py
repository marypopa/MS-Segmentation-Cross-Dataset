import torch
from torch.utils.data import Dataset as BaseDataset
import os
from PIL import Image

"""
    Make a flexible Dataset such as it can easily
    used with all datasets that follow the same
    structure

    attibutes:
    images_dirs: of type list containing directly the path to the data to be considered
    img_slices_folder_name: the name of the folder where are the slices stored for each patient
    seg_slices_folder_name: the name of the folder where are the segmentation slices stored for each patient
"""

def get_dataset(datasets, patient):
    datasets_list=datasets.split(',')
    for i in range(len(datasets_list)):
        if datasets_list[i] in patient:
            return i

class Dataset(BaseDataset):
    def __init__(self,
                 patients,
                 datasets,
                 images_folder,
                 segmentations,
                 transform=None
                 ):
        self.images_folder = images_folder
        self.transform = transform
        self.images_paths = []
        self.segmentation_paths = []

        for patient in patients:
            image_slices_folder_path = os.path.join(patient, self.images_folder)
            images = os.listdir(image_slices_folder_path)
            images_sorted = sorted(images, key=lambda x: int(x.split('_')[-1].split('.')[0]))

            patient_images = [os.path.join(image_slices_folder_path, image_path) for image_path in images_sorted]

            segmentations_list = segmentations.split(',')
            seg_slices_folder_path = os.path.join(patient, segmentations_list[get_dataset(datasets, patient)])
            masks = []
            if os.path.exists(seg_slices_folder_path):
                mask_images = os.listdir(seg_slices_folder_path)
                mask_sorted = sorted(mask_images, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                masks = [os.path.join(seg_slices_folder_path, seg_path) for seg_path in mask_sorted]
            self.images_paths = self.images_paths + patient_images
            self.segmentation_paths = self.segmentation_paths + masks

    def __getitem__(self, i):
        #print('Get item: ', i)
        image = Image.open(self.images_paths[i])
        if len(self.segmentation_paths) > 0:
            seg = Image.open(self.segmentation_paths[i])
            if self.transform:
                seg = self.transform(seg)
        else:
            seg = torch.zeros(image.size)
        if self.transform:
            image = self.transform(image)
        # return image.to(torch.float32), seg/65535
        return image.to(torch.float32), seg/65535, self.images_paths[i]  #return file names as third arg

    def __len__(self):
        return len(self.images_paths)

    def print_dataset(self):
        print('Dataset contains ' + str(len(self.images_paths)) + ' images and ' + str(
            len(self.segmentation_paths)) + ' segmentations')
