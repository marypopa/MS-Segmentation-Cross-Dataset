import os
import subprocess
def run_command(command):
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the command: {e}")

def n4BiasFieldCorrection(path, file):
    file_path = os.path.join(path, file)
    output_file = os.path.join(path, 'n4_'+file)
    run_command(['N4BiasFieldCorrection', "-d", "3", "-i", file_path, '-o', output_file])

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

def process_image(from_path, to_path, t1_image, flair_image, masks, template, brain_mask=None, n4Bias=None):
    original_flair_image = os.path.join(from_path, flair_image)
    if n4Bias:
        n4BiasFieldCorrection(from_path, flair_image)
        original_flair_image = os.path.join(from_path, 'n4_' + flair_image)
    original_t1_image = os.path.join(from_path, t1_image)
    reoriented_t1_image = os.path.join(to_path, 're_' + t1_image)
    reoriented_flair_image = os.path.join(to_path, 're_' + flair_image)
    registered_t1_image = os.path.join(to_path, 'MNI_'+t1_image)
    registered_flair_image = os.path.join(to_path, 'MNI_'+flair_image)
    mat_reg = os.path.join(to_path, 'MNI_'+t1_image+'.mat')
    mat_reorient = os.path.join(to_path, 're_'+t1_image+'.mat')

    ## Reorient T1, FLAIR and mask to MNI
    print('Reorient')
    run_command(['fslreorient2std', '-m', mat_reorient, original_flair_image, reoriented_flair_image])
    run_command(['fslreorient2std', original_t1_image, reoriented_t1_image])


    # Register FLAIR to MNI
    print('Registration to MNI')
    TWO_STEP_REGISTRATION = True
    if TWO_STEP_REGISTRATION:
        # 2 step
        # Register T1 to MNI and save .mat
        run_command(['flirt', '-in', reoriented_t1_image,'-ref', template, '-out', registered_t1_image, '-omat', mat_reg ])
        # apply mat to FLAIR
        run_command(['flirt', '-in', reoriented_flair_image, '-ref', template, '-out', registered_flair_image, '-applyxfm', '-init', mat_reg])
    else:
        # directly register FLAIR to MNI (perhaps less precise, because FLAIR and MNI have different contrasts - FLAIR vs T1)
        run_command(['flirt', '-in', reoriented_flair_image, '-ref', template, '-out', registered_flair_image, '-omat', mat_reg])

    if not isinstance(masks, list):
        masks = [masks]
    for mask in masks:
        original_mask = os.path.join(from_path, mask)
        reoriented_mask = os.path.join(to_path, 're_' + mask)
        registered_mask = os.path.join(to_path, 'MNI_' + mask)
        # run_command(['flirt', '-in', original_mask, '-ref', original_mask, '-out', reoriented_mask, '-applyxfm', '-init', mat_reorient])
        run_command(['fslreorient2std', original_mask, reoriented_mask])
        # Register mask to MNI
        print('Register mask to MNI')
        run_command(['flirt', '-in', reoriented_mask, '-ref', template, '-out', registered_mask, '-applyxfm', '-init', mat_reg])

    if brain_mask:
        original_brain_mask = os.path.join(from_path, brain_mask)
        reoriented_brain_mask = os.path.join(to_path, 're_' + brain_mask)
        registered_brain_mask = os.path.join(to_path, 'MNI_' + brain_mask)
        run_command(['flirt', '-in', original_brain_mask, '-ref', original_brain_mask, '-out', reoriented_brain_mask, '-applyxfm', '-init',  mat_reorient])
        run_command(['flirt', '-in', reoriented_brain_mask, '-ref', template, '-out', registered_brain_mask, '-applyxfm', '-init', mat_reg])

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

        # MNI_brain_mask = registered_mask


        registered_brain_mask_bin = os.path.join(to_path, 'bin_MNI_' + brain_mask)
        registered_flair_image_masked = os.path.join(to_path, 'masked_MNI_'+flair_image)
        final_flair = os.path.join(to_path, 'bet_MNI_'+flair_image)
        run_command(['fslmaths', registered_brain_mask, '-thr', '0.5', '-bin', registered_brain_mask_bin])
        # fslmaths input_image.nii.gz -mas mask_bin.nii.gz output_image.nii.gz
        run_command(['fslmaths', registered_flair_image, '-mas', registered_brain_mask_bin, registered_flair_image_masked])
        run_command(['bet', registered_flair_image_masked, final_flair, '-f', '0.2', '-m'])  # mask -> 'bet_MNI_'+flair_image+'_mask.nii.gz'

    else:
        final_flair = os.path.join(to_path, 'bet_MNI_' + flair_image)
        run_command(['bet', registered_flair_image, final_flair, '-f', '0.2', '-m']) # mask -> 'bet_MNI_'+flair_image + '_mask.nii.gz'