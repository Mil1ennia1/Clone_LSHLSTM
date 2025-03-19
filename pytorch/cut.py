def extract_lines_by_indices(file_path, indices, output_file_path):

    lines = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 确保indices是唯一的并且排序
    indices = sorted(set(indices))

    extracted_lines = [lines[i - 1] for i in indices if 0 < i <= len(lines)]

    with open(output_file_path, 'w') as outfile:
        outfile.writelines(extracted_lines)


def process_files(seq_file_path, codes_file_path, seq_output_path, codes_output_path):

    indices_to_extract = set(range(1, 501)) | set(range(7867202, 7867703))

    extract_lines_by_indices(seq_file_path, indices_to_extract, seq_output_path)

    extract_lines_by_indices(codes_file_path, indices_to_extract, codes_output_path)


seq_file_path = r"E:\pycharm\pycharm_library\neural_network\LSHLSTM\data\noisy_output.seq"  # 原始序列文件路径
codes_file_path = r"E:\pycharm\pycharm_library\neural_network\LSHLSTM\data\noisy_data_15"  # 编码值文件路径
seq_output_path = r"E:\pycharm\pycharm_library\neural_network\LSHLSTM\data\extracted_data.seq"  # 序列文件输出路径
codes_output_path = r"E:\pycharm\pycharm_library\neural_network\LSHLSTM\data\extracted_data15"  # 编码文件输出路径

process_files(seq_file_path, codes_file_path, seq_output_path, codes_output_path)