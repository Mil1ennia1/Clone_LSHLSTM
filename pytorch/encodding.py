def extract_label(id_part):
    try:
        label_value = int(id_part.split('|')[1])
        return 1 if label_value < 549 else 0
    except (IndexError, ValueError):
        print(f"警告：无法解析标签值 - {id_part}")
        return None


def replace_sequences_with_codes_and_labels(seq_file_path, codes_file_path, output_file_path):
    with open(seq_file_path, 'r') as seq_file, open(codes_file_path, 'r') as codes_file, open(output_file_path,
                                                                                              'w') as outfile:
        for seq_line in seq_file:
            code_line = codes_file.readline().strip()

            if not code_line:
                print("警告：编码文件中的行数不足")
                break

            parts = seq_line.strip().split('\t')
            if len(parts) < 3:  # 确保至少有三列
                print(f"警告：此行格式不正确 - {seq_line}")
                continue

            first_label = parts[0]
            id_part = parts[1]


            new_label = extract_label(id_part)
            if new_label is None:
                continue


            new_line = f"{first_label}\t{new_label}\t{code_line}\n"
            outfile.write(new_line)



seq_file_path = r"E:\pycharm\pycharm_library\neural_network\LSHLSTM\data\extracted_data.seq"
codes_file_path = r"E:\pycharm\pycharm_library\neural_network\LSHLSTM\data\extracted_data15"
output_file_path = r"E:\pycharm\pycharm_library\neural_network\LSHLSTM\data\NOISY15_encode.seq"

replace_sequences_with_codes_and_labels(seq_file_path, codes_file_path, output_file_path)