import numpy as np
from tqdm import tqdm


def load_embeddings(embedding_file_path):

    embeddings = {}
    with open(embedding_file_path, 'r') as file:

        total_vectors, vector_dim = map(int, file.readline().strip().split())

        for line in file:
            parts = line.strip().split()
            if len(parts) != vector_dim + 1:
                print(f"警告：此行格式不正确 - {line}")
                continue

            try:
                value = int(parts[0])
            except ValueError:

                continue

            vector = np.array([float(x) for x in parts[1:]])
            embeddings[value] = vector
    return embeddings


def process_line(line, embeddings):

    parts = line.strip().split('\t')
    if len(parts) < 3:
        print(f"警告：此行格式不正确 - {line}")
        return None

    new_label = parts[1]
    values = [int(x) for x in parts[2].strip().split()]


    embedding_matrix = []
    for value in values:
        if value in embeddings:
            embedding_matrix.append(embeddings[value])
        else:
            embedding_matrix.append(np.zeros(100))

    return new_label, np.array(embedding_matrix)


def save_to_npy(serial, new_label, matrix, output_dir):

    labels_file_path = f"{output_dir}/batch_{serial}_labels.npy"
    features_file_path = f"{output_dir}/batch_{serial}_features.npy"


    np.save(labels_file_path, np.array([new_label]))


    np.save(features_file_path, matrix)


def process_file(input_file_path, embedding_file_path, output_dir):
    embeddings = load_embeddings(embedding_file_path)


    with open(input_file_path, 'r') as infile:
        total_lines = sum(1 for _ in infile)

    with open(input_file_path, 'r') as infile:

        for line in tqdm(infile, total=total_lines, desc="Processing lines"):
            result = process_line(line, embeddings)
            if result is not None:
                new_label, embedding_matrix = result


                serial = line.strip().split('\t')[0]


                save_to_npy(serial, new_label, embedding_matrix, output_dir)


input_file_path = r"E:\pycharm\pycharm_library\neural_network\LSHLSTM\data\NOISY15_encode.seq"
embedding_file_path = r"E:\pycharm\pycharm_library\neural_network\LSHLSTM\data\LSHVEC_noisy_15.vec"
output_dir = r"E:\pycharm\pycharm_library\neural_network\LSHLSTM\data\train_data\cut_train_noisy_15mer"

process_file(input_file_path, embedding_file_path, output_dir)