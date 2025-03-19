import os
import re


def rename_files(folder_path):

    all_files = os.listdir(folder_path)

    pattern = re.compile(r'batch_(\d+)_([a-z]+)\.npy')


    file_pairs = {}


    for filename in all_files:
        match = pattern.match(filename)
        if match:
            original_index, file_type = match.groups()
            original_index = int(original_index)

            if original_index > 500:
                if original_index not in file_pairs:
                    file_pairs[original_index] = {'features': None, 'labels': None}

                file_pairs[original_index][file_type] = filename


    new_index = 501
    for original_index, files in file_pairs.items():
        if files['features'] and files['labels']:
            for file_type in ['features', 'labels']:
                old_name = files[file_type]
                old_path = os.path.join(folder_path, old_name)
                new_name = f'batch_{new_index}_{file_type}.npy'
                new_path = os.path.join(folder_path, new_name)


                if os.path.exists(new_path):
                    print(f'Skipping rename due to existing file: {old_name} -> {new_name}')
                    continue

                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                    print(f'Renamed: {old_name} -> {new_name}')


            new_index += 1



folder_path = r"E:\pycharm\pycharm_library\neural_network\LSHLSTM\data\train_data\cut_train_noisy_15mer"

rename_files(folder_path)