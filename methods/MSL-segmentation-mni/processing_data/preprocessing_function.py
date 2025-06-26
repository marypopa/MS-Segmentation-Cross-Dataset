import nibabel as nib
import numpy as np
import skimage.transform as skTrans
import os
import imageio
from torchvision import transforms
import shutil
from PIL import Image

transform = transforms.Compose([
            # you can add other transformations in this list
            transforms.ToTensor(),
            transforms.Resize((224,224))
        ])

def get_slices_index_not_totally_black(img_data):
    l = []
    for z in range(img_data.shape[2]):
        if not np.all(img_data[:,:,z] == 0):
            l.append(z)
    return l

def check_orientation(img):
    data = img.get_fdata()
    x, y, z = nib.aff2axcodes(img.affine)
    flip_L = False
    flip_A = False
    flip_S = False
    if x != 'L':
        # print('flip L')
        data = np.flip(data, axis=0)
        flip_L = True
    if y != 'A':
        # print('flip A')
        data = np.flip(data, axis=1)
        flip_A = True
    if z != 'S':
        # print('flip S')
        data = np.flip(data, axis=2)
        flip_S = True
    return data, flip_L, flip_A, flip_S

def apply_orientation(img, flip_L, flip_A, flip_S):
    data = img.get_fdata()
    x, y, z = nib.aff2axcodes(img.affine)
    if flip_L:
        # print('flip L')
        data = np.flip(data, axis=0)
        flip_L = True
    if flip_A:
        # print('flip A')
        data = np.flip(data, axis=1)
        flip_A = True
    if flip_S:
        # print('flip S')
        data = np.flip(data, axis=2)
        flip_S = True
    return data

def change_voxel_dimension(img_data, img):
    current_voxel_dims = np.abs(np.diagonal(img.affine)[:3])
    desired_voxel_dims = np.array([1.0, 1.0, img.header.get_zooms()[2]])
    scaling_factors = desired_voxel_dims / current_voxel_dims
    new_shape = [round(i) for i in img.shape/scaling_factors]
    result1 = skTrans.resize(img_data, new_shape, order=1, preserve_range=True)
    return result1

"""
    Split the MRI into png slices
    DATASET_PATH: dataset path
    datasetname: name to identify the dataset, used when creating the png name
    dirname: where to store the slices
    modality: name of the modality which appear after the patient
    segmentation: segmentation image name to be considered
"""
def split_image_into_png_slices(DATASET_PATH, datasetname, dirname, modality, dirsegmentation, segmentation, transform=None):
    for center in os.listdir(DATASET_PATH):
        center_path = os.path.join(DATASET_PATH, center)
        for patient in os.listdir(center_path):
            patient_path = os.path.join(center_path, patient)
            if not os.path.isdir(os.path.join(patient_path, dirname)):
                os.mkdir(os.path.join(patient_path, dirname))
            slices_directory = os.path.join(patient_path, dirname)

            if not os.path.isdir(os.path.join(patient_path, dirsegmentation)):
                os.mkdir(os.path.join(patient_path, dirsegmentation))
            segmentation_dir = os.path.join(patient_path, dirsegmentation)

            img_name = patient + '_'+modality +'.nii.gz'
            img = nib.load(os.path.join(patient_path, img_name))
            img_data = img.get_fdata()
            seg_name = patient + '_'+segmentation +'.nii.gz'
            seg = nib.load(os.path.join(patient_path, seg_name))
            seg_data = seg.get_fdata()

            transform_slices = []
            tranform_seg_slices = []

            for z in range(img_data.shape[2]):
                slice_name = datasetname+'_'+center+'_'+center+'_'+patient+'_'+ modality+'_'+str(z)+'.png'
                slice_path = os.path.join(slices_directory, slice_name)
                slice_uint8 = (img_data[:, :, z] * 255).astype(np.uint8)
                imageio.imwrite(slice_path, slice_uint8)

                if transform:
                    transform_slices.append(transform(slice_uint8))

                seg_slice_name = datasetname + '_' + center + '_' + center + '_' + patient + '_' + segmentation + '_' + str(
                    z) + '.png'
                seg_slice_path = os.path.join(segmentation_dir, seg_slice_name)
                seg_slice_uint8 = (seg_data[:, :, z] * 255).astype(np.uint8)
                imageio.imwrite(seg_slice_path, seg_slice_uint8)

                if transform:
                    tranform_seg_slices.append(transform(seg_slice_uint8))

            if len(transform_slices) > 0:
                image_path = os.path.join(patient_path, img_name)
                resized_image = np.stack([np.array(slice) for slice in transform_slices], axis=-1)
                nifti_img = nib.Nifti1Image(resized_image, affine=img.affine)
                nib.save(nifti_img, image_path.split('.nii', 1)[0] + '_resized.nii' + image_path.split('.nii', 1)[1])

                mask_path = os.path.join(patient_path, seg_name)
                resized_mask = np.stack([np.array(slice) for slice in tranform_seg_slices], axis=-1)
                nifti_img = nib.Nifti1Image(resized_mask, affine=seg.affine)
                nib.save(nifti_img, mask_path.split('.nii', 1)[0] + '_resized.nii' + mask_path.split('.nii', 1)[1])


"""
    normalization: choices: None (default), '3D', 'slice'
"""
def store_nifti_slices(img_data, img_name, folder, normalization, slices_index):
    if normalization == '3D':
        if np.max(img_data) != np.min(img_data):
            img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
        # Scale the normalized data to uint16 range [0, 65535]
        img_data = (img_data * 65535).astype(np.uint16)
    for z in slices_index:
        slice_name = img_name + '_' + str(z) + '.png'
        slice_path = os.path.join(folder, slice_name)
        slice = img_data[:, :, z]
        if normalization == 'slice':
            slice = (slice - np.min(slice)) / (np.max(slice) - np.min(slice))
            # Scale the normalized data to uint16 range [0, 65535]
            slice = (slice * 65535).astype(np.uint16)
        image = Image.fromarray((slice).astype(np.uint16))
        image.save(slice_path)
def remove_directory(DATASET_PATH,directory_name):
    for center in os.listdir(DATASET_PATH):
        center_path = os.path.join(DATASET_PATH, center)
        for train_test in os.listdir(center_path):
            train_test_path = os.path.join(center_path, train_test)
            for patient in os.listdir(train_test_path):
                patient_path = os.path.join(train_test_path, patient)
                if os.path.isdir(os.path.join(patient_path, directory_name)):
                    shutil.rmtree(os.path.join(patient_path, directory_name))
                    print('Directory removed for ', patient)

"""
   Extract the slices which contain at least one white pixel from the preproceesing flair  
"""
def select_slices(DATASET_PATH, datasetname):
    for center in os.listdir(DATASET_PATH):
        center_path = os.path.join(DATASET_PATH, center)
        for train_test in os.listdir(center_path):
            train_test_path = os.path.join(center_path, train_test)
            for patient in os.listdir(train_test_path):
                patient_path = os.path.join(train_test_path, patient)
                if not os.path.isdir(os.path.join(patient_path, 'selected_slices')):
                    os.mkdir(os.path.join(patient_path, 'selected_slices'))
                slices_directory = os.path.join(patient_path, 'selected_slices')
                mask_name = patient+'_FLAIR_raw_pp_bet_mask.nii.gz'
                mask = nib.load(os.path.join(patient_path, mask_name))
                mask_data = mask.get_fdata()
                img_name = patient+'_FLAIR_raw_pp_bet.nii.gz'
                img = nib.load(os.path.join(patient_path, img_name))
                img_data = img.get_fdata()
                for z in range(mask_data.shape[2]):
                    if np.any(mask_data[:, :, z] == 1):
                        slice_name = datasetname+'_'+center+'_'+train_test+'_'+patient+'_FLAIR_raw_pp_'+str(z)+'.png'
                        slice_path = os.path.join(slices_directory, slice_name)
                        slice_uint8 = (img_data[:, :, z] * 255).astype(np.uint8)
                        imageio.imwrite(slice_path, slice_uint8)