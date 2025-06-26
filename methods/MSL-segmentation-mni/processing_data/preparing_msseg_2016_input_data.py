import os
import nibabel as nib
import imageio
import numpy as np
import shutil
import skimage.transform as skTrans
from preprocessing_function import store_nifti_slices, check_orientation, get_slices_index_not_totally_black, apply_orientation
from quantile_normalization import quantile_normalization_3D_image
from process_image import *
from multiprocessing import Process
import argparse

template_name = 'MNI_TEMPLATE/MSSEG2016C8P3_bet_MNI_FLAIR_preprocessed'

FLAIR_TEMPLATE = nib.load(template_name+'.nii.gz').get_fdata()
FLAIR_BRAIN_MASK_TEMPLATE = nib.load(template_name+'_mask.nii.gz').get_fdata()
FLAIR_TEMPLATE = (FLAIR_TEMPLATE - np.min(FLAIR_TEMPLATE)) / (np.max(FLAIR_TEMPLATE) - np.min(FLAIR_TEMPLATE))*65535
template = 'MNI_TEMPLATE/MNI152_T1_1mm_brain.nii.gz'

parser = argparse.ArgumentParser(description='Standard processing')
parser.add_argument('-n', '--normalization', type=str, default='quantile', choices={'quantile', '3D'})
args = parser.parse_args()
def process_image_wrapper(args):
    process_image(*args)


# mask MNI_FLAIR with MNI_brain_mask(binarized) -> MNI_FLAIR_brain
# bet MNI_FLAIR_brain -f 0.2 -> MNI_FLAIR_brain_bet, BET_MASK
# quantile_norm(MNI_FLAIR_brain_bet, BET_MASK)
#
# Alternatively: bet MNI_FLAIR -f 0.2 -> MNI_FLAIR_bet
# quantile_norm(MNI_FLAIR_bet, MNI_brain_mask(binarized))
#
# For ISBI:
# bet MNI_FLAIR -f 0.? -> MNI_FLAIR_bet, BET_MASK
# quantile_norm(MNI_FLAIR_bet, BET_MASK)

### aplicam mask lui peste MNI si apoi inca o data bet cu 0.2
###
### iar pentru ISBI fiind ca nu are mask aplicam direct bet
###
processes = []
def process_images(working_path, dataset_name, path, normalization):
    dataset_path = os.path.join(path, dataset_name)
    print(dataset_path)
    processes = []
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    dirs = [folder for folder in os.listdir(working_path) if os.path.isdir(os.path.join(working_path, folder))]
    for center in dirs:
        if os.path.isdir(os.path.join(working_path, center)) and 'Center' in center:
            center_name = 'center' + center.split('_')[1]
            center_path = os.path.join(dataset_path, center_name)
            if not os.path.exists(center_path):
                os.mkdir(center_path)
            dirs_patient = [folder for folder in os.listdir(os.path.join(working_path, center)) if
                            os.path.isdir(os.path.join(working_path, center, folder))]
            for patient in dirs_patient:
                patient_initial_path = os.path.join(os.path.join(working_path, center), patient)
                new_patient_name = 'patient' + patient.split('_')[1]
                patient_path = os.path.join(center_path, new_patient_name)
                if not os.path.exists(patient_path):
                    os.mkdir(patient_path)

                processed_data_files = os.listdir(os.path.join(patient_initial_path, 'Preprocessed_Data'))
                masks_files = os.listdir(os.path.join(patient_initial_path, 'Masks'))
                processed_data_files_flair = [file for file in processed_data_files if 'FLAIR' in file]
                processed_data_files_t1 = [file for file in processed_data_files if 'T1' in file]
                processed_data_files_mask = [file for file in masks_files if 'Consensus' in file]

                brain_mask_file_name = [file for file in os.listdir(os.path.join(patient_initial_path, 'Masks')) if 'Brain_Mask' in file][0]
                brain_mask_file_path = os.path.join(os.path.join(patient_initial_path, 'Masks'), brain_mask_file_name)
                brain_mask = nib.load(brain_mask_file_path)

                shutil.copy(os.path.join(os.path.join(patient_initial_path, 'Masks'), brain_mask_file_name),
                            os.path.join(patient_initial_path, 'Preprocessed_Data', brain_mask_file_name))
                shutil.copy(os.path.join(os.path.join(patient_initial_path, 'Masks'), processed_data_files_mask[0]),
                            os.path.join(patient_initial_path, 'Preprocessed_Data', processed_data_files_mask[0]))
                t1_image = processed_data_files_t1[0]
                flair_image = processed_data_files_flair[0]
                mask = processed_data_files_mask[0]

                PARALLEL = True
                N_PROCESSES = 30
                N4BiasField = False
                if PARALLEL:
                    p = Process(target=process_image_wrapper, args=((os.path.join(patient_initial_path, 'Preprocessed_Data'), patient_path, t1_image,
                                                                     flair_image, mask, template, brain_mask_file_name,
                                                                     N4BiasField),))
                    p.start()
                    processes.append(p)
                    if len(processes) >= N_PROCESSES:
                        # Wait for all processes to finish
                        for p in processes:
                            p.join()
                        processes = []
                else:
                    process_image(os.path.join(patient_initial_path, 'Preprocessed_Data'), patient_path, t1_image, flair_image, mask, template,
                                  brain_mask_file_name, n4Bias=N4BiasField)

    for p in processes:
        p.join()
    processes = []

def create_files(working_path, dataset_name, path, normalization):
    dataset_path = os.path.join(path, dataset_name)
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    for center in os.listdir(dataset_path):
        center_path = os.path.join(dataset_path, center)
        for patient in os.listdir(center_path):
            patient_path = os.path.join(center_path, patient)
            processed_data_files = os.listdir(patient_path)
            processed_data_files_flair = [file for file in processed_data_files if 'bet_MNI' in file and 'FLAIR' in file]
            processed_data_files_mask = [file for file in processed_data_files if 'MNI' in file and 'Consensus' in file]
            brain_mask = [file for file in processed_data_files if 'bet_MNI' in file and '_mask' in file][0]

            flair_image = processed_data_files_flair[0]
            mask = processed_data_files_mask[0]

            file = flair_image
            new_file_name = patient + '_FLAIR_op.nii.gz'
            resized_name = patient + '_FLAIR_op_r.nii.gz'
            slices_folder = 'slices_FLAIR_op_r'
            slice_name = dataset_name + '_' + center + '_' + patient+'_FLAIR_op_r'
            shutil.copy(os.path.join(patient_path, file), os.path.join(patient_path, new_file_name))
            im = nib.load(os.path.join(patient_path, new_file_name))
            im_data = im.get_fdata()
            slices_index_not_fully_black = get_slices_index_not_totally_black(im_data)
            brain_mask_data = nib.load(os.path.join(patient_path, brain_mask)).get_fdata() > 0.5
            if normalization == 'quantile':
                im_data = quantile_normalization_3D_image(im_data, FLAIR_TEMPLATE, brain_mask_data, FLAIR_BRAIN_MASK_TEMPLATE, dataset_name+'_'+patient+'_'+center)

            nx = im_data.shape[0]
            ny = im_data.shape[1]
            nz = im_data.shape[2]
            # pad the image to multiples of 16/32 for input to Unet
            padded_img = np.zeros((192, 224, nz))
            padded_img[0:nx, 0:ny, :] = im_data
            nifti_img = nib.Nifti1Image(padded_img, affine=im.affine)
            nib.save(nifti_img, os.path.join(patient_path, resized_name))

            slices_folder_path = os.path.join(patient_path, slices_folder)
            if not os.path.exists(slices_folder_path):
                os.mkdir(slices_folder_path)
            store_nifti_slices(padded_img, slice_name, slices_folder_path, normalization, slices_index_not_fully_black)

            file = mask
            new_file_name = patient + '_seg.nii.gz'
            resized_name = patient + '_seg_r.nii.gz'
            slices_folder_path = os.path.join(patient_path, 'slices_seg_r')
            slice_name = dataset_name + '_' + center + '_' + patient + '_seg_r'

            shutil.copy(os.path.join(patient_path, file), os.path.join(patient_path, new_file_name))
            im = nib.load(os.path.join(patient_path, new_file_name))
            im_data = im.get_fdata()
            nx = im_data.shape[0]
            ny = im_data.shape[1]
            nz = im_data.shape[2]
            # pad the image to multiples of 16/32 for input to Unet
            padded_img = np.zeros((192, 224, nz))
            padded_img[0:nx, 0:ny, :] = im_data

            nifti_img = nib.Nifti1Image(padded_img, affine=im.affine)
            nib.save(nifti_img, os.path.join(patient_path, resized_name))

            if not os.path.exists(slices_folder_path):
                os.mkdir(slices_folder_path)

            store_nifti_slices(padded_img, slice_name, slices_folder_path, '3D', slices_index_not_fully_black)

dataset_path = '../../../Datasets/'
working_path_train = dataset_path+'MSSEG-2016/MSSEG-Training/Training'
path = dataset_path+'Preprocessed_mni_'+args.normalization
os.makedirs(path, exist_ok=True)
process_images(working_path_train, 'MSSEG_2016', path, args.normalization)
create_files(working_path_train, 'MSSEG_2016', path, args.normalization)
working_path_test = '../../../Datasets/MSSEG-2016/MSSEG-Testing/Testing'
process_images(working_path_test, 'MSSEG_2016_test', path, args.normalization)
create_files(working_path_test, 'MSSEG_2016_test', path, args.normalization)