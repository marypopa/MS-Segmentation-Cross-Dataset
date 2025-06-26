import os
import shutil
import nibabel as nib
import skimage.transform as skTrans
from preprocessing_function import store_nifti_slices, check_orientation, get_slices_index_not_totally_black, apply_orientation
from quantile_normalization import quantile_normalization_3D_image
import numpy as np
import subprocess
from process_image import *
import argparse
from multiprocessing import Process

template_name = 'MNI_TEMPLATE/MSSEG2016C8P3_bet_MNI_FLAIR_preprocessed'

FLAIR_TEMPLATE = nib.load(template_name+'.nii.gz').get_fdata()
FLAIR_BRAIN_MASK_TEMPLATE = nib.load(template_name+'_mask.nii.gz').get_fdata()
FLAIR_TEMPLATE = (FLAIR_TEMPLATE - np.min(FLAIR_TEMPLATE)) / (np.max(FLAIR_TEMPLATE) - np.min(FLAIR_TEMPLATE))*65535


parser = argparse.ArgumentParser(description='Standard processing')
parser.add_argument('-n', '--normalization', type=str, default='quantile', choices={'quantile', '3D'})
args = parser.parse_args()
template = 'MNI_TEMPLATE/MNI152_T1_1mm.nii.gz'

def run_command(command):
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the command: {e}")

def process_image_wrapper(args):
    process_image(*args)

def process_images(from_path, dataset_name, path, normalization):
    dataset_path = os.path.join(path, dataset_name)
    processes = []
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    center_path = os.path.join(dataset_path, 'center01')
    if not os.path.exists(center_path):
        os.mkdir(center_path)
    dirs = [folder for folder in os.listdir(from_path) if os.path.isdir(os.path.join(from_path, folder))]
    for folder in dirs:
        patient_path = os.path.join(center_path, folder)
        if not os.path.exists(patient_path):
            os.mkdir(patient_path)
        print('Processing patient - ', folder)
        patient_folder_path = os.path.join(from_path, folder)
        processed_data_files = os.listdir(patient_folder_path)
        # processed_data_files_without_flair = [file for file in processed_data_files if ('FLAIR_n4.nii.gz' not in file) and ('brainmask.nii.gz' not in file) and ('.nii.gz' in file)]
        processed_data_files_flair = [file for file in processed_data_files if 'FLAIR.nii.gz' in file and 'n4' not in file]
        processed_data_files_t1= [file for file in processed_data_files if 'T1W.nii.gz' in file]
        processed_data_files_mask = [file for file in processed_data_files if 'consensus' in file]
        # processed_data_files = processed_data_files_flair + processed_data_files_without_flair
        brain_mask_file_name = [file for file in os.listdir(os.path.join(patient_folder_path)) if 'brainmask.nii.gz' in file][0]

        t1_image = processed_data_files_t1[0]
        flair_image = processed_data_files_flair[0]
        mask = processed_data_files_mask[0]

        PARALLEL = True
        N_PROCESSES = 30
        N4BiasField = True
        if PARALLEL:
            p = Process(target=process_image_wrapper, args=((patient_folder_path, patient_path, t1_image, flair_image, mask, template, brain_mask_file_name, N4BiasField),))
            p.start()
            processes.append(p)
            if len(processes) >= N_PROCESSES:
                # Wait for all processes to finish
                for p in processes:
                    p.join()
                processes = []
        else:
            process_image(patient_folder_path, patient_path, t1_image, flair_image, mask, template, brain_mask_file_name, N4BiasField)
    for p in processes:
        p.join()
    processes = []

def create_files(from_path, dataset_name, path, normalization):
    dataset_path = os.path.join(path, dataset_name)
    center_path = os.path.join(dataset_path, 'center01')
    dirs = [folder for folder in os.listdir(center_path) if os.path.isdir(os.path.join(center_path, folder))]
    for folder in dirs:
        patient_path = os.path.join(center_path, folder)
        print('Processing patient - ', folder)
        processed_data_files = os.listdir(patient_path)
        # processed_data_files_without_flair = [file for file in processed_data_files if ('FLAIR_n4.nii.gz' not in file) and ('brainmask.nii.gz' not in file) and ('.nii.gz' in file)]
        processed_data_files_flair = [file for file in processed_data_files if 'bet_MNI' in file and 'FLAIR' in file]
        processed_data_files_mask = [file for file in processed_data_files if 'MNI' in file and 'consensus' in file]
        processed_data_files_brain_mask = [file for file in processed_data_files if 'bet_MNI' in file and '_mask' in file]

        flair_image = processed_data_files_flair[0]
        mask = processed_data_files_mask[0]
        brain_mask = processed_data_files_brain_mask[0]

        print('finished reorient and registration')
        modality = 'FLAIR'
        file = flair_image
        file_path = os.path.join(patient_path, file)

        new_file_name = folder + '_' + modality + '_op.nii.gz'
        resized_name = folder + '_' + modality + '_op_r.nii.gz'
        slices_folder = 'slices_' + modality + '_op_r'
        slice_name = dataset_name + '_center01_' + folder + '_' + modality + '_op_r'
        shutil.copy(file_path,
                    os.path.join(patient_path, new_file_name))
        im = nib.load(os.path.join(patient_path, new_file_name))
        if modality == 'FLAIR':
            im_data = im.get_fdata()
            slices_index_not_fully_black = get_slices_index_not_totally_black(im_data)
            brain_mask_data = nib.load(os.path.join(patient_path, brain_mask)).get_fdata()
            brain_mask_data_th = brain_mask_data > 0.5
            im_data = im_data * brain_mask_data_th
            if normalization == 'quantile':
                im_data = quantile_normalization_3D_image(im_data, FLAIR_TEMPLATE, brain_mask_data_th,
                                                      FLAIR_BRAIN_MASK_TEMPLATE,
                                                      dataset_name + '_' + folder + '_center01')
        nx = im_data.shape[0]
        ny = im_data.shape[1]
        nz = im_data.shape[2]
        # pad the image to multiples of 16/32 for input to Unet
        padded_img = np.zeros((192,224, nz))
        padded_img[0:nx, 0:ny, :] = im_data
        # result1 = skTrans.resize(im_data, (224, 224, z_shape), order=0, preserve_range=True, anti_aliasing=False)
        nifti_img = nib.Nifti1Image(padded_img, affine=im.affine)
        nib.save(nifti_img, os.path.join(patient_path, resized_name))
        slices_folder_path = os.path.join(patient_path, slices_folder)
        if not os.path.exists(slices_folder_path):
            os.mkdir(slices_folder_path)
        store_nifti_slices(padded_img, slice_name, slices_folder_path, normalization, slices_index_not_fully_black)

        #process mask
        modality = 'seg'
        file_path = os.path.join(patient_path, mask)
        new_file_name = folder + '_' + modality + '.nii.gz'
        resized_name = folder + '_' + modality + '_r.nii.gz'
        slices_folder = 'slices_' + modality + '_r'
        slice_name = dataset_name + '_center01_' + folder + '_' + modality + '_r'
        shutil.copy(file_path,
                    os.path.join(patient_path, new_file_name))
        im = nib.load(os.path.join(patient_path, new_file_name))
        im_data = im.get_fdata()
        nx = im_data.shape[0]
        ny = im_data.shape[1]
        nz = im_data.shape[2]
        padded_img = np.zeros((192, 224, nz))
        padded_img[0:nx, 0:ny, :] = im_data
        # result1 = skTrans.resize(im_data, (224, 224, z_shape), order=1, preserve_range=True,anti_aliasing=False)
        nifti_img = nib.Nifti1Image(padded_img, affine=im.affine)
        nib.save(nifti_img, os.path.join(patient_path, resized_name))
        slices_folder_path = os.path.join(patient_path, slices_folder)
        if not os.path.exists(slices_folder_path):
            os.mkdir(slices_folder_path)
        store_nifti_slices(padded_img, slice_name, slices_folder_path, '3D', slices_index_not_fully_black)

def run(from_path, dataset_name, save_path, normalization):
    process_images(from_path, dataset_name, save_path, normalization)
    create_files(from_path, dataset_name, save_path, normalization)

dataset_path = '../../../Datasets/'
from_path = dataset_path+'3D-MR-MS/patients'
dataset_name = '3D_MR_MS'
save_path = dataset_path+'Preprocessed_mni_'+args.normalization
os.makedirs(save_path, exist_ok=True)
run(from_path, dataset_name, save_path, args.normalization)

## ce e acum salvat in "Processed_datasets_normalized_quantile_MSSEGTrainC8P3" pentru 3D_MR_MS este varianta cu N4BiasFieldCorrection
