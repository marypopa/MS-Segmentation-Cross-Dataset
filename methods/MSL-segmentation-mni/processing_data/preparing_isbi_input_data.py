import os
import shutil
import nibabel as nib
import skimage.transform as skTrans
from preprocessing_function import store_nifti_slices, get_slices_index_not_totally_black, check_orientation, apply_orientation
import numpy as np
from quantile_normalization import quantile_normalization_3D_image
from process_image import *
from multiprocessing import Process
import argparse

template_name = 'MNI_TEMPLATE/MSSEG2016C8P3_bet_MNI_FLAIR_preprocessed'

FLAIR_TEMPLATE = nib.load(template_name+'.nii.gz').get_fdata()
FLAIR_BRAIN_MASK_TEMPLATE = nib.load(template_name+'_mask.nii.gz').get_fdata()
FLAIR_TEMPLATE = (FLAIR_TEMPLATE - np.min(FLAIR_TEMPLATE)) / (np.max(FLAIR_TEMPLATE) - np.min(FLAIR_TEMPLATE))*65535

parser = argparse.ArgumentParser(description='Standard processing')
parser.add_argument('-n', '--normalization', type=str, default='quantile', choices={'quantile', '3D'})
args = parser.parse_args()
template = 'MNI_TEMPLATE/MNI152_T1_1mm_brain.nii.gz'
def process_image_wrapper(args):
    process_image(*args)


def process_images(from_path, dataset_name, path, normalization):
    processes = []
    dataset_path = os.path.join(path, dataset_name)
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    if not os.path.exists(os.path.join(dataset_path, 'center01')):
        os.mkdir(os.path.join(dataset_path, 'center01'))
    working_path = os.path.join(dataset_path, 'center01')
    for folder in os.listdir(from_path):
        folder_path = os.path.join(from_path, folder)
        if os.path.isdir(folder_path):
            preprocessed_data_files = os.listdir(os.path.join(folder_path, 'preprocessed'))
            preprocessed_data_files_without_flair = [file for file in preprocessed_data_files if 'flair' not in file]
            preprocessed_data_files_flair = [file for file in preprocessed_data_files if 'flair' in file and 'brain_mask' not in file]
            preprocessed_data_files = preprocessed_data_files_flair + preprocessed_data_files_without_flair

            for file in preprocessed_data_files_flair:
                patient_name = 'patient'+folder[len(folder)-2:]+'@'+file.split('_')[1]
                patient_path = os.path.join(working_path, patient_name)
                if not os.path.exists(patient_path):
                    os.mkdir(patient_path)

                t1_image = file.split('flair')[0] + 'mprage'+file.split('flair')[1]
                flair_image = file
                masks_files =  [ f for f in os.listdir(os.path.join(folder_path, 'masks')) if file.split('_flair')[0] in f]
                for mask in masks_files:
                    shutil.copy(os.path.join(folder_path, 'masks', mask), os.path.join(folder_path, 'preprocessed', mask))
                masks_files.sort()
                #create consensus mask in original space
                mask1 = nib.load(os.path.join(folder_path, 'masks', masks_files[0]))
                mask1_data = mask1.get_fdata()
                mask2 = nib.load(os.path.join(folder_path, 'masks', masks_files[1]))
                mask2_data = mask2.get_fdata()
                consensus_mask = mask1_data + mask2_data
                consensus_mask[consensus_mask >1 ] = 1
                nifti_img = nib.Nifti1Image(consensus_mask, affine=mask1.affine)
                nib.save(nifti_img, os.path.join(folder_path, 'preprocessed', flair_image.split('flair')[0]+'Consensus.nii.gz'))
                masks_files.append(flair_image.split('flair')[0]+'Consensus.nii.gz')


                PARALLEL = True
                N_PROCESSES = 30
                N4BiasField = False
                BrainMask = None
                if PARALLEL:
                    p = Process(target=process_image_wrapper,
                                args=((os.path.join(folder_path, 'preprocessed'), patient_path, t1_image,
                                       flair_image, masks_files, template, BrainMask,
                                       N4BiasField),))
                    p.start()
                    processes.append(p)
                    if len(processes) >= N_PROCESSES:
                        # Wait for all processes to finish
                        for p in processes:
                            p.join()
                        processes = []
                else:
                    process_image(os.path.join(folder_path, 'preprocessed'), patient_path, t1_image,
                                  flair_image, masks_files, template,
                                  BrainMask, n4Bias=N4BiasField)
    for p in processes:
        p.join()
    processes = []

def create_files(from_path, dataset_name, path, normalization):
    dataset_path = os.path.join(path, dataset_name)

    working_path = os.path.join(dataset_path, 'center01')
    for folder in os.listdir(working_path):
        folder_path = os.path.join(working_path, folder)
        preprocessed_data_files = os.listdir(folder_path)
        flair = [file for file in preprocessed_data_files if 'bet_MNI' in file and 'flair_pp' in file][0]
        masks = [file for file in preprocessed_data_files if 'MNI' in file and 'flair_brain_mask' not in file and 'mask' in file]
        consensus = [file for file in preprocessed_data_files if 'MNI' in file and 'Consensus' in file][0]
        slices_folder_name = 'slices_FLAIR_op_r'
        slice_name = 'center01_' + folder + '_FLAIR_op_r'
        new_file_name = folder + '_FLAIR_op.' + flair.split('.', 1)[1]
        resized_name = folder + '_FLAIR_op_r.' + flair.split('.', 1)[1]

        shutil.copy(os.path.join(folder_path, flair), os.path.join(folder_path, new_file_name))
        im = nib.load(os.path.join(folder_path, new_file_name))
        im_data = im.get_fdata()
        slices_index_not_fully_black = get_slices_index_not_totally_black(im.get_fdata())

        brain_mask_data = np.zeros(im_data.shape)
        brain_mask_data[np.where(im_data > 0)] = 1

        if normalization == 'quantile':
            im_data = quantile_normalization_3D_image(im_data, FLAIR_TEMPLATE, brain_mask_data,
                                                      FLAIR_BRAIN_MASK_TEMPLATE,
                                                      dataset_name + '_' + folder + '_center01')

        nx = im_data.shape[0]
        ny = im_data.shape[1]
        nz = im_data.shape[2]
        # pad the image to multiples of 16/32 for input to Unet
        padded_img = np.zeros((192, 224, nz))
        padded_img[0:nx, 0:ny, :] = im_data
        nifti_img = nib.Nifti1Image(padded_img, affine=im.affine)
        nib.save(nifti_img, os.path.join(folder_path, resized_name))
        slices_folder_path = os.path.join(folder_path, slices_folder_name)
        if not os.path.exists(slices_folder_path):
            os.mkdir(slices_folder_path)
        store_nifti_slices(padded_img, slice_name, slices_folder_path, normalization, slices_index_not_fully_black)

        for file in masks:
            new_file_name = folder + '_manseg_' +file.split('.')[0][len(file.split('.')[0])-1] + '.'+file.split('.',1)[1]
            resized_name = folder + '_manseg_r_' +file.split('.')[0][len(file.split('.')[0])-1] + '.'+file.split('.',1)[1]
            slice_name = 'ISBI2015_center01_'+folder+'_manseg_r_'+file.split('.')[0][len(file.split('.')[0])-1]
            shutil.copy(os.path.join(folder_path, file), os.path.join(folder_path, new_file_name))
            im = nib.load(os.path.join(folder_path, new_file_name))
            im_data = im.get_fdata()
            nx = im_data.shape[0]
            ny = im_data.shape[1]
            nz = im_data.shape[2]
            # pad the image to multiples of 16/32 for input to Unet
            padded_img = np.zeros((192, 224, nz))
            padded_img[0:nx, 0:ny, :] = im_data
            nifti_img = nib.Nifti1Image(padded_img, affine=im.affine)
            nib.save(nifti_img, os.path.join(folder_path, resized_name))

            slices_folder_name = 'slices_manseg_r_'+file.split('.')[0][len(file.split('.')[0])-1]
            slices_folder_path = os.path.join(folder_path, slices_folder_name)
            if not os.path.exists(slices_folder_path):
                os.mkdir(slices_folder_path)
            store_nifti_slices(padded_img, slice_name, slices_folder_path, '3D', slices_index_not_fully_black)

        # process consensus mask

        im = nib.load(os.path.join(folder_path, consensus))
        im_data = im.get_fdata()

        new_file_name = folder + '_seg_' + file.split('.')[0][len(file.split('.')[0]) - 1] + '.' + \
                        file.split('.', 1)[1]
        resized_name = folder + '_seg_r_' + file.split('.')[0][len(file.split('.')[0]) - 1] + '.' + \
                       file.split('.', 1)[1]
        slice_name = 'ISBI2015_center01_' + folder + '_seg_r_' + file.split('.')[0][
            len(file.split('.')[0]) - 1]


        nx = im_data.shape[0]
        ny = im_data.shape[1]
        nz = im_data.shape[2]
        # pad the image to multiples of 16/32 for input to Unet
        padded_img = np.zeros((192, 224, nz))
        padded_img[0:nx, 0:ny, :] = im_data
        nifti_img = nib.Nifti1Image(padded_img, affine=im.affine)
        nib.save(nifti_img, os.path.join(folder_path, resized_name))

        slices_folder_name = 'slices_seg_r'
        slices_folder_path = os.path.join(folder_path, slices_folder_name)
        if not os.path.exists(slices_folder_path):
            os.mkdir(slices_folder_path)
        store_nifti_slices(padded_img, slice_name, slices_folder_path, '3D', slices_index_not_fully_black)

dataset_path = '../../../Datasets/'
path = dataset_path+'ISBI_2015/training_final_v4/training/'
directory_path = dataset_path+'Preprocessed_mni_'+args.normalization
os.makedirs(directory_path, exist_ok=True)
process_images(path,'ISBI_2015',directory_path, args.normalization)
create_files(path,'ISBI_2015',directory_path, args.normalization)
