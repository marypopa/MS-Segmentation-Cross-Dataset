import os

def get_unique_filename(filename):
    new_file_name = filename
    index = 1
    while os.path.exists(new_file_name+'_best_dice_score.pth.tar'):
        new_file_name = filename+'_'+str(index)
        index = index+1
    return new_file_name
def get_patients(data_path, datasets_name):
    patients = []
    datasets = datasets_name.split(',')
    for i in range(len(datasets)):
        dataset_path = os.path.join(data_path, datasets[i])
        for center in os.listdir(dataset_path):
            center_path = os.path.join(dataset_path, center)
            for patient in os.listdir(center_path):
                patient_path = os.path.join(center_path, patient)
                patients.append(patient_path)
    return patients

def get_datasets_data(data_path, datasets_name, segmentations, images):
    datasets = datasets_name.split(',')
    segmentation_folders = segmentations.split(segmentations)
    images_folders = images.split(images)
    im_file_names = []
    seg_file_names = []
    seg_file_paths = []
    im_file_paths = []
    for i in range(len(datasets)):
        dataset_path = os.path.join(data_path, datasets[i])
        seg_path = os.path.join(dataset_path, segmentation_folders[i])
        images_path = os.path.join(dataset_path, images_folders[i])
        images_file_name = [file for file in os.listdir(images_path)]
        images_file_name.sort()
        images_file_path = [os.path.join(images_path, file) for file in images_file_name]
        seg_name = [file for file in os.listdir(seg_path)]
        seg_name.sort()
        seg_file_path = [os.path.join(seg_path, file) for file in seg_name]
        im_file_names  = im_file_names + images_file_name
        seg_file_names = seg_file_names + seg_name
        im_file_paths = im_file_paths + images_file_path
        seg_file_paths = seg_file_paths + seg_file_path
    return {
        'images_name': im_file_names,
        'images_path': im_file_paths,
        'seg_name': seg_file_names,
        'seg_path': seg_file_paths
    }


def split_list(lst, n, k):
    result = []
    for i in range(0, len(lst), k):
        result.append(lst[i:i+k])
    last_part = result[-1]
    while len(last_part) < k:
        last_part.append(result.pop())
    return result

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
from collections import defaultdict

def generate_plot(data, epochs, args=None):
    filename = get_unique_filename(args.model_name)
    fig = plt.figure(1, figsize=(5,5))
    plt.clf()
    field_values = defaultdict(list)
    for d in data:
        for key, value in d.items():
            field_values[key].append(value)

    # Convert defaultdict to a normal dictionary (optional)
    field_values = dict(field_values)

    n = 0
    n_keys = len(field_values)
    sorted_items = list(field_values.items())
    sorted_items.sort()
    for i in range(0, n_keys, 2):
        n = n + 1
        plt.subplot(n_keys // 2, 1, i // 2 + 1)
        plt.plot(epochs, sorted_items[i][1], label=sorted_items[i][0], color='blue')
        plt.plot(epochs, sorted_items[i + 1][1], label=sorted_items[i + 1][0], color='red')
        if i == n_keys-2:
            plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title(sorted_items[i][0] + ' and ' + sorted_items[i + 1][0])
        plt.legend()
        # Try to maximize the window in a backend-independent way
        # manager = plt.get_current_fig_manager()
        # try:
        #     manager.window.state('zoomed')  # Works for TkAgg (Windows/Linux)
        # except AttributeError:
        #     try:
        #         manager.full_screen_toggle()  # Works for Qt5Agg, MacOS
        #     except AttributeError:
        #         fig.set_size_inches(15, 10)  # Fallback: manually increase size
        plt.gcf().set_size_inches(15, 10)
        if args:
            plt.savefig(os.path.join(args.path_test, 'train '+filename, args.model_name + '_' + str(args.seed) + '.png'), dpi=300, bbox_inches='tight')
        else:
            plt.savefig('training_plot.png', dpi=300, bbox_inches='tight')