from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchsummary import summary
from numpy import ndarray, dtype, unsignedinteger
from numpy._typing import _16Bit
import numpy as np
import pandas as pd
import random
from PIL import Image
import time
import os
import nibabel as nib
import argparse
import shutil
import scipy.ndimage
import cc3d

import models.my_smp.segmentation_models_pytorch as smp
from models.my_smp.segmentation_models_pytorch import utils
from models.my_smp.segmentation_models_pytorch.utils.meter import AverageValueMeter
from models.my_smp.segmentation_models_pytorch.utils.my_utils import WeightedBCELoss
from models.dataset import Dataset
from models.utils import get_patients, get_unique_filename, generate_plot



MODELS = {'Unet', 'UnetPlusPlus', 'MAnet', 'Linknet', 'FPN', 'PSPNet', 'PAN', 'DeepLabV3', 'DeepLabV3Plus'}

parser = argparse.ArgumentParser(description='Multiple Sclerosis Lesions Segmentation')
parser.add_argument('-m', '--model', metavar='MODEL', default='UnetPlusPlus', choices=MODELS,
                    help='model architecture: ' + ' | '.join(MODELS) + ' (default: UnetPlusPlus)')
parser.add_argument('-e', '--encoder', metavar='ENCODER', default='resnet18',
                    help='encoder architecture: https://smp.readthedocs.io/en/latest/encoders_timm.html (default: resnet18)')
parser.add_argument('--optimizer', metavar='OPTIMIZER', default='adam', choices={'sgd','adam','adamw'},
                    help='optimizer: sgd or adam or adamw (default: adam)')
parser.add_argument('-a', '--activation', metavar='ACTIVATION', default='sigmoid',
                    help='activation function (default: sigmoid)')
parser.add_argument('--loss', metavar='LOSS', default='weighted_bce', choices={'dice','iou','bce','weighted_bce'},
                    help='loss function: {dice,iou,bce,weighted_bce} (default: weighted_bce)')
parser.add_argument('--weighted-bce-weight', dest='weighted_bce_weight', metavar='wBCEw', default=0.8, type=float,
                    help='weight used in weighted_bce (default: 0.8)')
parser.add_argument('-w', '--num_workers', default=24, type=int, metavar='N',
                    help='number of data loading workers (default: 24)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=80, type=int, metavar='N',
                    dest='batch_size', help='batch size')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-step-size', default=20, type=int,
                    metavar='LR_STEP_SIZE', help='learning rate step size (default: 20)', dest='lr_step_size')
parser.add_argument('--lr-factor', default=0.5, type=float,
                    metavar='LR_FACTOR', help='learning rate factor (default: 0.5)', dest='lr_factor')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training (default None, i.e. random seed). ')
parser.add_argument('--slice-size', dest='slice_size', default=(224, 224), type=tuple, metavar='(N,N)',
                    help='slice image dimensions, default=(224, 224)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', dest='resume',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-t', '--training', dest='training', action='store_true',
                    help='model training mode')
parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False,
                    help='shuffle training data in dataloader, default: False')
parser.add_argument('-i', '--inference', dest='inference', action='store_true',
                    help='model inference mode')
parser.add_argument('--threshold', default=0.5, type=float, metavar='THR',
                    help='threshold used for training and inference (default: 0.5)')
parser.add_argument('--model-name', dest='model_name', metavar='STR', default='',
                    help='path to model or where to store the model')
parser.add_argument('--data-path', default='', type=str, metavar='PATH', dest='data_path',
                    help='path to the dataset (default: current directory)')
parser.add_argument('--datasets', default='MSSEG_2016', type=str, metavar='N', dest='datasets',
                    help='datasets name comma separated (default: "MSSEG_2016")')
parser.add_argument('--segmentations', default='slices_seg_r', type=str, metavar='N', dest='segmentations',
                    help='seg mask folders used in datasets as mask comma separated (default: "slices_seg_r")')
parser.add_argument('--images', default='slices_FLAIR_op_r', type=str, metavar='N', dest='images',
                    help='images slices folders used in datasets as mask comma separated (default: "slices_FLAIR_op_r")')
parser.add_argument('--val-ratio', default='0.2', type=float, metavar='N', dest='val_ratio',
                    help='training validation split, validation ration (default: 0.2)')
parser.add_argument('--save-predictions', dest='store', action='store_true', default=False,
                    help='shuffle training data in dataloader, default: False')
parser.add_argument('--save-predictions-path', default='', type=str, metavar='PATH', dest='path_test',
                    help='path where to store predictions')
parser.add_argument('--save-model-freq', default=20, type=int, metavar='N', dest='save_model_freq',
                    help='save the model every n epochs')
parser.add_argument('--results-file-name', dest='results_file_name', metavar='STR', default='results_quality_annotation_isbi_quantile_c8p3_adam.xls',
                    help='file to save the results')
parser.add_argument('--scheduler', metavar='SCHEDULER', default='StepLR', choices={'StepLR','PolynomialLR'},
                    help='scheduler: StepLR or PolynomialLR (default: StepLR)')
parser.add_argument('--model-trained-on', dest='trained_on', metavar='STR', default='',
                    help='the dataset name the model was trained on')


STRUCTURE18 = np.array([[[0, 1, 0],
  [1, 1, 1],
  [0, 1, 0]],
 [[1, 1, 1],
  [1, 1, 1],
  [1, 1, 1]],
 [[0, 1, 0],
  [1, 1, 1],
  [0, 1, 0]]])
STRUCTURE = STRUCTURE18

def _lesion_f1_score(truth, prediction, empty_value=1.0):
    """
    Computes the lesion-wise F1-score between two masks. Masks are considered true positives if at least one voxel
    overlaps between the truth and the prediction.

    Parameters
    ----------
    truth : array-like, bool
        3D array. If not boolean, will be converted.
    prediction : array-like, bool
        3D array with a shape matching 'truth'. If not boolean, will be converted.
    empty_value : scalar, float
        Optional. Value to which to default if there are no labels. Default: 1.0.

    Returns
    -------
    f1_score : float
        Lesion-wise F1-score as float.
        Max score = 1
        Min score = 0
        If both images are empty (tp + fp + fn =0) = empty_value

    Notes
    -----
    This function computes lesion-wise score by defining true positive lesions (tp), false positive lesions (fp) and
    false negative lesions (fn) using 3D connected-component-analysis.

    tp: 3D connected-component from the ground-truth image that overlaps at least on one voxel with the prediction image.
    fp: 3D connected-component from the prediction image that has no voxel overlapping with the ground-truth image.
    fn: 3d connected-component from the ground-truth image that has no voxel overlapping with the prediction image.
    """
    tp, fp, fn = 0, 0, 0
    f1_score = empty_value

    labeled_ground_truth, num_lesions = scipy.ndimage.label(truth.astype(bool), structure=STRUCTURE)

    # For each true lesion, check if there is at least one overlapping voxel. This determines true positives and
    # false negatives (unpredicted lesions)
    for idx_lesion in range(1, num_lesions+1):
        lesion = (labeled_ground_truth == idx_lesion) + 0.
        lesion_pred_sum = lesion + prediction
        if(np.max(lesion_pred_sum) > 1):
            tp += 1
        else:
            fn += 1

    # For each predicted lesion, check if there is at least one overlapping voxel in the ground truth.
    labaled_prediction, num_pred_lesions = scipy.ndimage.label(prediction.astype(bool)+0., structure=STRUCTURE)
    print('number of lesions', num_lesions, 'number predicted lesions', num_pred_lesions)

    for idx_lesion in range(1, num_pred_lesions+1):
        lesion = (labaled_prediction == idx_lesion) + 0.
        lesion_pred_sum = lesion + truth
        if(np.max(lesion_pred_sum) <= 1):  # No overlap
           fp += 1

    # Compute f1_score
    denom = tp + (fp + fn)/2
    if(denom != 0):
        f1_score = tp / denom
    return f1_score


def lesion_f1_score(truth, prediction, batchwise=False):
    """ Computes the F1 score lesionwise. Lesions are considered accurately predicted if a single voxel overlaps between
    a region in `truth` and `prediction`.

    Parameters
    ----------
    truth : array-like, bool
        Array containing the ground truth of a sample, of shape (channel, x, y, z). Returned F1 score is the mean
        across the channels. If batchwise=True, array should be 5D with (batch, channel, x, y, z).
    prediction : array-like, bool
        Array containing predictions for a sample; description is otherwise identical to `truth`.
    batchwise : bool
        Optional. Indicate whether the computation should be done batchwise, assuming that the first dimension of the
        data is the batch. Default: False.

    Returns
    -------
    float or tuple
        Lesion-wise F1-score. If batchwise=True, the tuple is the F1-score for every sample.
    """
    if not batchwise:
        num_channel = truth.shape[0]
        f1_score = _lesion_f1_score(truth[0, ...], prediction[0, ...])
        for i in range(1, num_channel):
            f1_score += _lesion_f1_score(truth[i, ...], prediction[i, ...])
        return f1_score / num_channel
    else:
        f1_list = []
        num_batch = truth.shape[0]
        for idx_batch in range(num_batch):
            f1_list.append(lesion_f1_score(truth[idx_batch, ...], prediction[idx_batch, ...], batchwise=False))
        return f1_list

def compute_lesion_f1_score(ground_truth, prediction, empty_value=1.0, connectivity=18):
    """
    Computes the lesion-wise F1-score between two masks.

    Parameters
    ----------
    ground_truth : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    prediction : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
    empty_value : scalar, float.
    connectivity : scalar, int.

    Returns
    -------
    f1_score : float
        Lesion-wise F1-score as float.
        Max score = 1
        Min score = 0
        If both images are empty (tp + fp + fn =0) = empty_value

    Notes
    -----
    This function computes lesion-wise score by defining true positive lesions (tp), false positive lesions (fp) and
    false negative lesions (fn) using 3D connected-component-analysis.

    tp: 3D connected-component from the ground-truth image that overlaps at least on one voxel with the prediction image.
    fp: 3D connected-component from the prediction image that has no voxel overlapping with the ground-truth image.
    fn: 3d connected-component from the ground-truth image that has no voxel overlapping with the prediction image.
    """
    ground_truth = np.asarray(ground_truth).astype(bool)
    prediction = np.asarray(prediction).astype(bool)
    tp = 0
    fp = 0
    fn = 0

    # Check if ground-truth connected-components are detected or missed (tp and fn respectively).
    intersection = np.logical_and(ground_truth, prediction)
    labeled_ground_truth, N = cc3d.connected_components(
        ground_truth, connectivity=connectivity, return_N=True
    )

    # Iterate over ground_truth clusters to find tp and fn.
    # tp and fn are only computed if the ground-truth is not empty.
    if N > 0:
        for _, binary_cluster_image in cc3d.each(labeled_ground_truth, binary=True, in_place=True):
            if np.logical_and(binary_cluster_image, intersection).any():
                tp += 1
            else:
                fn += 1

    # iterate over prediction clusters to find fp.
    # fp are only computed if the prediction image is not empty.
    labeled_prediction, N = cc3d.connected_components(
        prediction, connectivity=connectivity, return_N=True
    )
    if N > 0:
        for _, binary_cluster_image in cc3d.each(labeled_prediction, binary=True, in_place=True):
            if not np.logical_and(binary_cluster_image, ground_truth).any():
                fp += 1

    # Define case when both images are empty.
    if tp + fp + fn == 0:
        _, N = cc3d.connected_components(ground_truth, connectivity=connectivity, return_N=True)
        if N == 0:
            f1_score = empty_value
    else:
        f1_score = tp / (tp + (fp + fn) / 2)

    return f1_score

args = parser.parse_args()
def main():

    print(args)
    if args.seed is None:
        random_seed = torch.randint(low=0, high=100000, size=(1,)).item()
    else:
        random_seed = args.seed
    print('random_seed = ' + str(random_seed))

    # Set the seed for CPU random number generator
    torch.manual_seed(random_seed)
    # Set the seed for GPU (if available) random number generator
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor(),
        # transforms.Resize(args.slice_size)
    ])

    metrics = [
        utils.metrics.IoU(threshold=args.threshold),
        utils.metrics.Dice(threshold=args.threshold),
        utils.metrics.FP(threshold=args.threshold),
        utils.metrics.FN(threshold=args.threshold),
        utils.metrics.TP(threshold=args.threshold),
    ]

    if args.training:
        filename = get_unique_filename(args.model_name)
        model_path = './'
        if args.path_test:
            model_path = args.path_test
        model_path = os.path.join(model_path, 'train '+ filename)
        os.makedirs(model_path, exist_ok=True)

        model = eval('smp.'+args.model)(
            encoder_name=args.encoder,
            in_channels=1,
            activation='sigmoid',
            # encoder_weights="imagenet"
        )
        print(model)
        # print model summary
        summary(model.to('cuda'), input_size=(1, args.slice_size[0], args.slice_size[1]))

        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)  # use all available GPUs

        print('Dataset path: ' + args.data_path)
        print('Datasets: ' + args.datasets)
        print('Images: ' + args.images)
        print('Seg: ' + args.segmentations)

        patients = get_patients(args.data_path, args.datasets)
        # Shuffle the patients list in order to mix the centers
        patients = [patients[i] for i in torch.randperm(len(patients))]

        print('Validation split: ', args.val_ratio)
        val_ratio = args.val_ratio
        split = int(np.floor(val_ratio * len(patients)))
        print('Patients considered for validation: ', split)
        training_patients = patients[split:]
        validation_patients = patients[:split]

        training_data = Dataset(
            training_patients,
            args.datasets,
            args.images,
            args.segmentations,
            transform
        )

        validation_data = Dataset(
            validation_patients,
            args.datasets,
            args.images,
            args.segmentations,
            transform
        )

        print('Training examples: ', len(training_data))
        print('Validation examples: ', len(validation_data))
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Model parameters: ', num_params)
        print('batch_size = ' + str(args.batch_size))

        # epochs = 100
        print('epochs = ' + str(args.epochs))

        def worker_init_fn(worker_id):
            np.random.seed(random_seed + worker_id)


        train_loader = torch.utils.data.DataLoader(training_data, batch_size=args.batch_size, shuffle=args.shuffle, pin_memory=True, num_workers=args.num_workers, worker_init_fn=worker_init_fn)

        valid_loader = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers, worker_init_fn=worker_init_fn)

        if args.loss == 'dice':
            loss_function_name = 'dice_loss'
            loss = utils.losses.DiceLoss()
            #loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif args.loss == 'iou':
            loss_function_name = 'jaccard_loss'
            loss = utils.losses.JaccardLoss()
        elif args.loss == 'bce':
            loss_function_name = 'bce_loss'
            # We are using BCELoss rather than BCEWithLogitsLoss since network output is in [0,1]
            loss = utils.losses.BCELoss()
        elif args.loss == 'weighted_bce':
            loss_function_name = 'weighted_bce_loss'
            s = []
            for x, y, _ in train_loader:
                s.append(y.mean())
            wy = 1 - sum(s) / len(s)
            # w = 0.8
            w = args.weighted_bce_weight
            loss = WeightedBCELoss(w)

        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        """Sets the learning rate to the initial LR decayed by LR_FACTOR every LR_STEP_SIZE epochs"""

        if args.scheduler == 'StepLR':
            scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_factor)
        elif args.scheduler == 'PolynomialLR':
            scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.epochs, power=0.9, last_epoch=-1, verbose=True)  #last_epoch to set to resume epoch????

        train_epoch = utils.train.TrainEpoch(
            model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            verbose=True,
            device='cuda'
        )

        valid_epoch = utils.train.ValidEpoch(
            model,
            loss=loss,
            metrics=metrics,
            verbose=True,
            device='cuda'
        )

        max_iou_score = 0
        max_dice_score = 0
        min_loss = float("inf")
        max_iou_score_dataset = 0
        max_dice_patient = 0

        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                loc = 'cuda'
                checkpoint = torch.load(args.resume, map_location=loc, weights_only=False)
                args.start_epoch = checkpoint['epoch'] + 1
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch'] + 1))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        all_train_logs = []
        all_valid_logs = []
        epochs = []
        graphic_data = []
        for epoch in range(args.start_epoch, args.epochs):

            print('\nEpoch: {}'.format(epoch))
            train_logs = train_epoch.run(train_loader)
            all_train_logs.append(train_logs)
            valid_logs = valid_epoch.run(valid_loader)
            all_valid_logs.append(valid_logs)
            print('train: ', train_logs, '\nval: ', valid_logs)

            #display graphic
            graphic_data.append({'loss_train': train_logs[args.loss+'_loss'], 'loss_val': valid_logs[args.loss+'_loss'], 'dice_train': train_logs['dice_score'], 'dice_val':valid_logs['dice_score'], 'dice_patient_train': train_logs['dice_patient_train'], 'dice_patient_val': valid_logs['dice_patient_valid']})
            epochs.append(epoch)
            generate_plot(graphic_data, epochs, args)

            for param_group in optimizer.param_groups:
                print("Current learning rate : {}".format(param_group['lr']))
            scheduler.step()

            model_state = {
                'args': args,
                'model': args.model,
                'encoder': args.encoder,
                'activation': args.activation,
                'traning_patients': training_patients,
                'validation_patients': validation_patients,
                'batch_size': args.batch_size,
                'epoch': epoch,
                'seed': random_seed,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_logs': train_logs,
                'valid_logs': valid_logs,
                'all_train_logs': all_train_logs,
                'all_valid_logs': all_valid_logs,
                'loss': loss,
            }

            if epoch % args.save_model_freq == 0 or epoch == (args.epochs-1):
                torch.save(model_state, os.path.join(model_path, filename + '_'+ str(random_seed) +'_' + str(epoch) + '.pth.tar'))
                print('Just saved model!')

            # save best model w.r.t. current loss function
            if min_loss > valid_logs[loss_function_name]:
                min_loss = valid_logs[loss_function_name]
                torch.save(model_state, os.path.join(model_path, filename + '_best_' + loss_function_name + '_'+ str(random_seed)+ '.pth.tar'))
                print('Best model saved!')
            # save best model w.r.t. IoU score
            if max_iou_score < valid_logs['iou_score']:
                max_iou_score = valid_logs['iou_score']
                torch.save(model_state, os.path.join(model_path, filename + '_best_iou_score_'+str(random_seed)+'.pth.tar'))
                print('Best IoU model saved!')
            # save best model w.r.t. Dice score
            if max_dice_score < valid_logs['dice_score']:
                max_dice_score = valid_logs['dice_score']
                torch.save(model_state, os.path.join(model_path, filename + '_best_dice_score_'+str(random_seed)+'.pth.tar'))
                print('Best Dice model saved!')
            # save best model w.r.t. IoU score on dataset
            if max_iou_score_dataset < valid_logs['iou_dataset_valid']:
                max_iou_score_dataset = valid_logs['iou_dataset_valid']
                torch.save(model_state, os.path.join(model_path, filename + '_best_iou_valid_dataset_'+str(random_seed)+'.pth.tar'))
                print('Best dataset IoU model saved!')
            # best dataset Dice model coincides with best dataset IoU model and thus need not be saved
            if max_dice_patient < valid_logs['dice_patient_valid']:
                max_dice_patient = valid_logs['dice_patient_valid']
                torch.save(model_state, os.path.join(model_path, filename + '_best_dice_patient_'+str(random_seed)+'.pth.tar'))
                print('Best Dice patient model saved!')


    if args.inference and args.trained_on:
        path = args.path_test
        # path_test = os.path.join(path, args.datasets + "_" + args.model_name + "_AdamCPP3")
        path_test = os.path.join(path, 'train '+args.trained_on, 'test ' + args.datasets)

        if not os.path.exists(path_test):
            os.mkdir(path_test)
        model_file = torch.load(os.path.join(path, 'train ' + args.trained_on, args.model_name + '.pth.tar'), weights_only=False)
        model = eval('smp.' + model_file['model'])(
            encoder_name=model_file['encoder'],
            in_channels=1,
            activation=model_file['activation']
        )

        # Data parallelism
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)  # use all available GPUs

        model.load_state_dict(model_file['state_dict'])
        model = model.to('cuda:0') #after moving the model to cuda 0 residual memory on cuda 1 remains, see more about pin_memory

        testing_patients = get_patients(args.data_path, args.datasets)

        df = pd.DataFrame(columns=['Dataset', 'Center', 'Patient', 'IoU', 'Dice', 'F1', 'Path'])

        with torch.no_grad():
            model.eval()
            iou_over_scans = []
            dice_over_scans = []
            lesion_wise_f1_scans = []
            # lesion_wise_f1_scans_cc3d = []
            for patient in testing_patients:
                print(patient)
                testing_data = Dataset(
                    [patient],
                    args.datasets,
                    args.images,
                    args.segmentations,
                    transform
                )

                img_name = args.images.split('_',1)[1]
                test_loader = torch.utils.data.DataLoader(testing_data, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

                pred_batch = []
                y_batch = []
                x_batch = []
                logs = {}
                dataset_indices_batch = []  # To store dataset indices for each batch
                metrics_meters = {metric.__name__: AverageValueMeter() for metric in metrics}
                for x, y, _ in test_loader:
                    pred = model(x)
                    pred_batch.append(pred.to('cpu'))
                    y_batch.append(y.to('cpu'))
                    x_batch.append(x.to('cpu'))


                    # update metrics logs

                    for metric_fn in metrics:
                        metric_value = metric_fn(pred.to('cpu'), y.to('cpu')).detach().cpu()

                        metrics_meters[metric_fn.__name__].add(metric_value)
                    metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                    logs.update(metrics_logs)

                iou_scan = logs["tp"] / (logs["tp"] + logs["fp"] + logs["fn"])
                dice_scan = 2 * logs["tp"] / (2 * logs["tp"] + logs["fp"] + logs["fn"])
                iou_over_scans.append(iou_scan)
                dice_over_scans.append(dice_scan)

                # compute lesion-wise Dice / F1
                pred_n = torch.cat(pred_batch, dim=0).numpy()  #pred_n[z,c,x,y]
                y_n = torch.cat(y_batch, dim=0).numpy()
                pred_3D = pred_n.transpose((1,2,3,0))  #pred_3D[c,x,y,z]
                pred_3D_threshold = pred_3D >= args.threshold
                y_3D = y_n.transpose((1,2,3,0))
                y_3D_threshold = y_3D >= args.threshold


                # lesion_wise_f1_scan = lesion_f1_score(y_3D_threshold, pred_3D_threshold)
                lesion_wise_f1_scan = compute_lesion_f1_score(np.squeeze(y_3D_threshold), np.squeeze(pred_3D_threshold))
                lesion_wise_f1_scans.append(lesion_wise_f1_scan)
                # lesion_wise_f1_scan_cc3d = compute_lesion_f1_score(np.squeeze(y_3D_threshold), np.squeeze(pred_3D_threshold))
                # lesion_wise_f1_scans_cc3d.append(lesion_wise_f1_scan_cc3d)
                print('lesion F1', lesion_wise_f1_scan)
                # print('lesion F1 cc3D', lesion_wise_f1_scan_cc3d)

                patient_path = patient.split('/')
                df_1 = pd.DataFrame({
                    'Dataset': [patient_path[len(patient_path)-3]],
                    'Center': [patient_path[len(patient_path)-2]],
                    'Patient': [patient_path[len(patient_path)-1]],
                    'IoU': [iou_scan],
                    'Dice': [dice_scan],
                    'F1': [lesion_wise_f1_scan],
                    # 'F1 cc3D': [lesion_wise_f1_scan_cc3d],
                    'Path': [patient],
                })
                df = pd.concat([df, df_1], ignore_index=True)

                center = patient_path[len(patient_path)-2]
                dir = os.path.join(path_test, center + '_' + patient_path[len(patient_path)-1])
                if not os.path.exists(dir):
                    os.mkdir(dir)
                if args.store:
                    # pred_n = torch.cat(pred_batch, dim=0).numpy()
                    # y_n = torch.cat(y_batch, dim=0).numpy()
                    x_n = torch.cat(x_batch, dim=0).numpy()
                    x_n /= 65535
                    x_3D = x_n.transpose((2, 3, 0, 1))

                    # save the overlap of the predicted slices with the ground truth in a separate dir in the patient directory
                    directory_path = os.path.join(dir, args.model_name + '_pred_1')
                    directory_brain_path = os.path.join(dir, args.model_name + '_pred_brain')
                    if not os.path.exists(directory_path):
                        os.mkdir(directory_path)
                    if not os.path.exists(directory_brain_path):
                        os.mkdir(directory_brain_path)
                    #save nifti
                    nifti_img = nib.Nifti1Image(np.squeeze(pred_3D_threshold) + 0.0, affine=np.eye(4))
                    nib.save(nifti_img, os.path.join(dir, 'pred.nii.gz'))
                    nifti_img = nib.Nifti1Image(np.squeeze(x_3D), affine=np.eye(4))
                    nib.save(nifti_img, os.path.join(dir, 'flair.nii.gz'))
                    nifti_img = nib.Nifti1Image(np.squeeze(y_3D_threshold)+0.0, affine=np.eye(4))
                    nib.save(nifti_img, os.path.join(dir, 'seg.nii.gz'))

                    # inters = pred_n * y_n
                    # pred_y = pred_n * (1-y_n)
                    # y_pred = (1-pred_n)*y_n

                    img = np.zeros((y_n.shape[0],3,y_n.shape[2],y_n.shape[3]))
                    # predictions will be shown in RED (channel 0)
                    # the ground truth in BLUE (channel 2)
                    # so that the correct predictions (overlapping with the ground truth) will appear in MAGENTA=RED+BLUE
                    img[:,0,:,:] = np.float32(pred_n[:,0,:,:] > args.threshold)
                    # img[:,0,:,:] = pred_n[:,0,:,:]
                    img[:,2,:,:] = y_n[:,0,:,:]
                    # img16 = (img*65535).astype(np.uint16)
                    img8 = (img*255).astype(np.uint8)  # the predictions
                    # construct a 3d color image of tp,fp,fn
                    pred3d = pred_n[:, 0, :, :] > args.threshold
                    gt3d = y_n[:,0,:,:] > args.threshold
                    fp = pred3d & ~gt3d
                    tp = pred3d & gt3d
                    fn = (~pred3d) & gt3d
                    FP_VALUE = 1.0   #magenta on Cool colormap from fsleyes
                    TP_VALUE = 0.5   #blue
                    FN_VALUE = 0.10  #cyan
                    color4d = np.zeros(tp.shape)
                    color4d[fp] = FP_VALUE
                    color4d[tp] = TP_VALUE
                    color4d[fn] = FN_VALUE

                    # show the predictions in color surrounded by the original grayscale brain
                    img_brain = np.copy(img)
                    ind = np.where(img_brain.sum(1, keepdims=True) < 1e-7)

                    img_brain[ind[0], 0, ind[2], ind[3]] = x_n[ind[0], 0, ind[2], ind[3]]
                    img_brain[ind[0], 1, ind[2], ind[3]] = x_n[ind[0], 0, ind[2], ind[3]]
                    img_brain[ind[0], 2, ind[2], ind[3]] = x_n[ind[0], 0, ind[2], ind[3]]
                    print(np.max(img_brain))

                    # img_brain16 = (img_brain * 65535).astype(np.uint16)
                    img_brain8 = (img_brain * 255).astype(np.uint8)

                    for i in range(img8.shape[0]):

                        image = Image.fromarray(np.transpose(img8[i], (1,2,0)))  #doesn't work for uint16, only for uint8!
                        image.save(os.path.join(directory_path, f"{i:04}"+'.png'))
                        image_brain = Image.fromarray(np.transpose(img_brain8[i], (1, 2, 0)))
                        image_brain.save(os.path.join(directory_brain_path, f"{i:04}" + '.png'))

                    #save colored nifti
                    color_nifti = color4d.transpose((1,2,0))
                    nifti_img = nib.Nifti1Image(color_nifti, affine=np.eye(4))
                    nib.save(nifti_img, os.path.join(dir,  'colored_pred.nii.gz'))
                    # y_img = nib.Nifti1Image(np.squeeze(y_3D), affine=np.eye(4))
                    # nib.save(y_img, os.path.join(directory_path, 'yNifti.nii.gz'))

            print('iou average over scans: ', sum(iou_over_scans)/len(iou_over_scans))
            print('dice average over scans: ', sum(dice_over_scans) / len(dice_over_scans))
            print('lesionwise F1 average over scans: ', sum(lesion_wise_f1_scans) / len(lesion_wise_f1_scans))
            # print('lesionwise F1 average over scans cc3d: ', sum(lesion_wise_f1_scans_cc3d) / len(lesion_wise_f1_scans_cc3d))
            results_file = os.path.join(args.path_test, args.results_file_name)
            if not os.path.exists(results_file):
                with open(results_file, "w") as file:
                    print("Train\tTest\tDice\tIoU\tF1", file=file)
            with open(results_file, "a") as file:
                # Print a string, directing the output to the opened file
                print(args.model_name,"\t", args.datasets,"\t", sum(dice_over_scans) / len(dice_over_scans),"\t", sum(iou_over_scans) / len(iou_over_scans), "\t", sum(lesion_wise_f1_scans) / len(lesion_wise_f1_scans), file=file)

            df.to_csv(os.path.join(path_test, args.model + '_test.tsv'), sep='\t', index=False)

if __name__ == '__main__':
    main()
