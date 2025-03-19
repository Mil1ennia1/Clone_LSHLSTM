import random


def add_sequence_noise(input_file, output_file, noise_ratio=0.2, replace_chars='ATCG'):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 3:
                outfile.write(line + '\n')
                continue

            original_seq = parts[2]
            seq_len = len(original_seq)

            if seq_len == 0:
                outfile.write(line + '\n')
                continue


            replace_num = max(1, min(round(noise_ratio * seq_len), seq_len))


            indices = random.sample(range(seq_len), replace_num)


            seq_list = list(original_seq)
            for idx in indices:
                seq_list[idx] = random.choice(replace_chars)

            parts[2] = ''.join(seq_list)
            outfile.write('\t'.join(parts) + '\n')


add_sequence_noise(r"E:\pycharm\pycharm_library\neural_network\LSHLSTM\data\data.seq", 'noisy_output.seq', replace_chars='ATCG')