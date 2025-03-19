import numpy as np


file_path = (r"E:\pycharm\pycharm_library\neural_network\LSHLSTM\data\train_data\cut_train_data_13mer\batch_101_features.npy")


data = np.load(file_path)


print("Array shape:", data.shape)
print((data))