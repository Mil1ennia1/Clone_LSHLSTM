
import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # 用于显示进度条
import logging
import random
import re
import numpy as np


_LOGGERS = {}

def get_logger(name):
    if name not in _LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        _LOGGERS[name] = logger
    return _LOGGERS[name]

logger = get_logger("hashSeq")


BASE_MAP = {'A': -1, 'T': 1, 'C': -1j, 'G': 1j}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RandProjTorch(torch.nn.Module):
    def __init__(self, kmer_size, hash_size):
        super().__init__()
        self.kmer_size = kmer_size
        self.hash_size = hash_size
        self.proj = torch.nn.Linear(kmer_size * 2, hash_size, bias=False)
        torch.nn.init.normal_(self.proj.weight)

    def forward(self, x):
        return (self.proj(x) > 0).float()

    def hash_read_batch(self, kmers_batch):
        batch_hashes = []
        for kmers in kmers_batch:
            vectors = []
            for kmer in kmers:
                vec = []
                for base in kmer:
                    real_part = BASE_MAP[base].real
                    imag_part = BASE_MAP[base].imag
                    vec += [real_part, imag_part]
                vectors.append(torch.tensor(vec, dtype=torch.float32))

            if not vectors:
                batch_hashes.append(torch.zeros(self.hash_size))
                continue

            kmer_tensor = torch.stack(vectors).to(DEVICE)
            hashed = self.forward(kmer_tensor)
            batch_hashes.append(hashed.mean(dim=0))

        return torch.stack(batch_hashes)


def extract_label(label_str):

    match = re.search(r'label\|(\d+)\|', label_str)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"无法解析标签: {label_str}")

def generate_kmer_for_fastseq(seq, k):

    return [
        ''.join([random.choice('ATCG') if c == 'N' else c
                 for c in seq[i:i + k]])
        for i in range(len(seq) - k + 1)
    ]

class SequenceDataset(Dataset):
    def __init__(self, filename, k):
        self.filename = filename
        self.k = k
        self.offsets = []
        self.seq_ids = []
        self.labels = []
        with open(filename, 'r') as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                parts = line.strip().split('\t')
                if len(parts) == 3 and parts[2].strip():
                    seq_id, label_str, seq = parts
                    seq = seq.upper()
                    if all(c in 'ATGCN' for c in seq):
                        try:
                            self.offsets.append(pos)
                            self.seq_ids.append(int(seq_id))
                            self.labels.append(extract_label(label_str))
                        except:
                            continue
        self.num_samples = len(self.offsets)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with open(self.filename, 'r') as f:
            f.seek(self.offsets[idx])
            line = f.readline().strip()
            _, _, seq = line.split('\t')
            seq = seq.upper()
            kmers = generate_kmer_for_fastseq(seq, self.k)
            return self.seq_ids[idx], self.labels[idx], seq, kmers


def collate_fn(batch, rp, bucket):
    seq_ids, labels, seqs, kmers_list = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)


    all_kmers = []
    sample_indices = []
    current = 0
    for kmers in kmers_list:
        all_kmers.extend(kmers)
        sample_indices.append((current, current + len(kmers)))
        current += len(kmers)

    if not all_kmers:
        hashes = torch.zeros(len(kmers_list), rp.hash_size).to(DEVICE)
    else:

        kmer_size = rp.kmer_size
        real_map = {'A': -1, 'T': 1, 'C': 0, 'G': 0}
        imag_map = {'A': 0, 'T': 0, 'C': -1, 'G': 1}
        kmers_array = np.array([list(kmer) for kmer in all_kmers])
        real = np.vectorize(real_map.get)(kmers_array)
        imag = np.vectorize(imag_map.get)(kmers_array)
        vectors = np.stack([real, imag], axis=-1).reshape(len(all_kmers), -1)
        kmer_tensor = torch.tensor(vectors, dtype=torch.float32).to(DEVICE)


        with torch.no_grad():
            hashed = rp(kmer_tensor)

        hashes = []
        for start, end in sample_indices:
            if start >= end:
                avg_hash = torch.zeros(rp.hash_size, device=DEVICE)
            else:
                avg_hash = hashed[start:end].mean(dim=0)
            hashes.append(avg_hash)
        hashes = torch.stack(hashes)

    hashes = (hashes * bucket).long() % bucket
    return seq_ids, labels, seqs, hashes

def convert_torch(in_file, out_file, hash_fun, kmer_size, n_thread,
                  hash_size, batch_size, bucket, create_lsh_only, lsh_file):
    logger.info("初始化转换流程...")


    rp = None
    if hash_fun == 'lsh':
        if lsh_file and os.path.exists(lsh_file):
            logger.info(f"加载LSH模型: {lsh_file}")
            rp = RandProjTorch(kmer_size, hash_size)
            rp.load_state_dict(torch.load(lsh_file))
            rp.to(DEVICE)
        else:
            logger.info("创建新LSH模型...")
            rp = RandProjTorch(kmer_size, hash_size).to(DEVICE)
            torch.save(rp.state_dict(), f"{out_file}.lsh.pth")
            if create_lsh_only:
                return

    # 准备数据
    dataset = SequenceDataset(in_file, kmer_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_thread if os.name != 'nt' else 0,
        collate_fn=lambda b: collate_fn(b, rp, bucket)
    )

    # 处理数据
    with open(out_file, 'w') as fout:
        fout.write("serial,label,hash\n")
        for seq_ids, labels, seqs, hashes in tqdm(dataloader, desc="Processing", unit="batch"):
            for seq_id, label, seq, hash_val in zip(seq_ids, labels, seqs, hashes):
                try:

                    fout.write(f"{seq_id},{label},{' '.join(map(str, hash_val.tolist()))}\n")
                except Exception as e:
                    logger.error(f"处理序列 {seq_id} 时出错: {str(e)}")
                    continue

    logger.info("转换完成")


if __name__ == "__main__":

    random.seed(42)
    torch.manual_seed(42)


    params = {
        'in_file': r"E:\pycharm\pycharm_library\neural_network\LSHLSTM\data\test.seq",
        'out_file': r"E:\pycharm\pycharm_library\neural_network\LSHLSTM\data\test_hash_15mer.csv",
        'hash_fun': 'lsh',
        'kmer_size': 15,
        'hash_size': 22,
        'bucket': 20000000,
        'n_thread': 4,
        'batch_size':32,
        'lsh_file': None,
        'device': 'cpu',
    }

    if not params['in_file'] or not os.path.exists(params['in_file']):
        logger.error("必须指定有效输入文件")
        sys.exit(1)


    DEVICE = torch.device(params['device'])
    logger.info(f"运行设备: {DEVICE}")


    convert_torch(
        in_file=params['in_file'],
        out_file=params['out_file'],
        hash_fun=params['hash_fun'],
        kmer_size=params['kmer_size'],
        n_thread=params['n_thread'],
        hash_size=params['hash_size'],
        batch_size=params['batch_size'],
        bucket=params['bucket'],
        create_lsh_only=False,
        lsh_file=params['lsh_file'],
    )
