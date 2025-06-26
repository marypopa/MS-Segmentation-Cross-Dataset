import os
import subprocess
import sys

'''
List of tasks in the form:
{ 
    "Training datasets":["Test dataset1", "Test dataset2", ...],
    "Training datasets":["testonly", "Test dataset1", "Test dataset2", ...],  # this only tests the model trained on "Training datasets" on the respective tests datasets
    ...
}
'''
tasks = {
    "MSSEG_2016_test": ["MSSEG_2016","3D_MR_MS","ISBI_2015"],
    "MSSEG_2016,3D_MR_MS,ISBI_2015": ["MSSEG_2016_test"],
}

default_params = {
    "model": "UnetPlusPlus",
    "encoder": "vgg16_bn",
    "segmentations": "slices_seg_r",
    "batch_size": "40",
    "seed": "36",
    "optimizer": "adam",
    "wd": "5e-7",
    "loss": "weighted_bce",
    "weighted_bce_weight": "0.8",
    "epochs": "50",
    "lr": "5e-4",
    "lr_step_size": "20",
    "lr_factor": "0.5",
    "workers": "36",
    "data_path": "../../Datasets/Preprocessed_mni_quantile/",
    "save_predictions_path": "./results_vgg16_quantile",
    "batch_size_test": "95",
}
BEST_CRITERION = 'best_dice_score'

def is_debug_mode():
    return sys.gettrace() is not None

def run_model(command, log_filename):
    try:
        if is_debug_mode():
            # direct subprocess call if in debug mode
            subprocess.run(command, check=True)
        else:
            # shell=True for enabling tee
            command.append(' | tee ' + '\"' + log_filename + '\"')
            command_str = ' '.join(command)
            subprocess.run(command_str, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the command: {e}")

def run_command(command):
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the command: {e}")

# Common settings
python_path = "python"
script_path = "models.run_model"

class IdentityDict:
   def __getitem__(self,key):
     return key
model_file = IdentityDict()



for train_datasets, test_datasets in tasks.items():
    command = [python_path, "-m", script_path, "-m", default_params["model"],
               "--encoder", default_params["encoder"], "--data-path", default_params["data_path"],
               "--segmentations", default_params["segmentations"], "--model-name", model_file[train_datasets],
               "--datasets", train_datasets, "--seed", default_params["seed"],
               "--save-predictions-path", '"' + default_params["save_predictions_path"] + '"']
    command_train = command + ["-t",
            "--epochs", default_params["epochs"], "--lr", default_params["lr"],
            "--lr-step-size", default_params["lr_step_size"], "--lr-factor", default_params["lr_factor"],
            "--batch-size", default_params["batch_size"], "--loss", default_params["loss"],
            "--weighted-bce-weight", default_params["weighted_bce_weight"], "--wd", default_params["wd"],
            "-w", default_params["workers"], "--optimizer", default_params["optimizer"], "--shuffle"]
    training_datasets_count = train_datasets.count(',') + 1
    if training_datasets_count > 1:
        segmentations = ','.join(["slices_seg_r"] * training_datasets_count)
        command_train = command_train + ["--segmentations", segmentations]

    if len(test_datasets) > 0 and test_datasets[0] == "testonly":
        test_datasets = test_datasets[1:]
    else:
        print('train: ', command_train)
        run_model(command_train, default_params["save_predictions_path"] + "/train " + model_file[train_datasets] + ".log")
        run_command(['mv', default_params["save_predictions_path"] + "/train " + model_file[train_datasets] + ".log", default_params["save_predictions_path"] + "/train " + model_file[train_datasets] + "/train " + model_file[train_datasets] + ".log"])

    for dataset in test_datasets:
        command_inference = command + ["-i",
            "--batch-size", default_params["batch_size_test"], "--save-predictions",
            "--save-predictions-path", '"'+default_params["save_predictions_path"] +'"',
            "--model-name", model_file[train_datasets]+"_"+BEST_CRITERION+"_"+default_params["seed"],
            "--model-trained-on", model_file[train_datasets],
            "--datasets", dataset,
            "--segmentations", "slices_seg_r",
            # "--results-file-name", "\'test " + model_file[train_datasets] + " on " + model_file[dataset] + ".xls\'"
            "--results-file-name", "results.xls"
        ]
        print(command_inference)
        run_model(command_inference,
                  default_params["save_predictions_path"] + "/train " + model_file[train_datasets] + "/test_" +
                  model_file[dataset] + ".log")
        run_command(['mv', default_params["save_predictions_path"] + "/train " + model_file[train_datasets] + "/test_" +
                     model_file[dataset] + ".log",
                     default_params["save_predictions_path"] + "/train " + model_file[train_datasets] + "/test " +
                     model_file[dataset] + "/test_" + model_file[dataset] + ".log"])
