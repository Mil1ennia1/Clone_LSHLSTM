import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
import mmap
from tqdm import tqdm
import os


class WindowsSafeDataset(Dataset):
    def __init__(self, csv_file, embedding_dict):
        self.csv_path = csv_file
        self.embedding = embedding_dict
        self.vec_dim = len(next(iter(embedding_dict.values())))
        self.offsets = self._preprocess_offsets()

    def _preprocess_offsets(self):

        offsets = [0]
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            f.readline()
            pbar = tqdm(desc="Calculating offsets", unit="rows")
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                offsets.append(pos)
                pbar.update(1)
            pbar.close()
        return offsets

    def _init_mmap(self):

        if not hasattr(self, '_mm'):
            self._file_handle = open(self.csv_path, 'r', encoding='utf-8')
            self._mm = mmap.mmap(
                self._file_handle.fileno(),
                0,
                access=mmap.ACCESS_READ
            )

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, idx):
        self._init_mmap()

        self._mm.seek(self.offsets[idx + 1])
        line = self._mm.readline().decode('utf-8').strip()
        row = list(csv.reader([line]))[0]


        feature_str = row[2]
        features = []
        for word in feature_str.split():
            features.append(
                self.embedding.get(
                    word,
                    np.zeros(self.vec_dim, dtype=np.float32)
                )
            )

        concatenated = np.concatenate(features) if features else np.zeros(self.vec_dim, dtype=np.float32)
        return torch.from_numpy(concatenated), torch.tensor(int(row[1]), dtype=torch.long)

    def __del__(self):
        if hasattr(self, '_mm'):
            self._mm.close()
        if hasattr(self, '_file_handle'):
            self._file_handle.close()


def windows_collate(batch):
    max_len = max([x[0].shape[0] for x in batch])
    padded = []
    labels = []
    for feat, label in batch:
        if feat.shape[0] < max_len:
            pad = torch.zeros(max_len - feat.shape[0], dtype=feat.dtype)
            feat = torch.cat([feat, pad])
        padded.append(feat)
        labels.append(label)
    return torch.stack(padded), torch.stack(labels)


def create_windows_loader(csv_path, emb_path, batch_size=64):
    # 流式加载嵌入
    embedding = {}
    with open(emb_path, 'r', encoding='utf-8') as f:
        next(f)  # 跳过头部
        for line in tqdm(f, desc="Loading embeddings", unit="vec"):
            parts = line.strip().split()
            embedding[parts[0]] = np.array(
                [float(x) for x in parts[1:]],
                dtype=np.float32
            )

    return DataLoader(
        WindowsSafeDataset(csv_path, embedding),
        batch_size=batch_size,
        collate_fn=windows_collate,
        num_workers=0,
        pin_memory=True
    )


if __name__ == "__main__":
    EMB_PATH = r"E:\pycharm\pycharm_library\neural_network\LSHLSTM\LSHVEC\15mer_vectors.vec"
    CSV_PATH = r"E:\pycharm\pycharm_library\neural_network\LSHLSTM\data\RawData\cutdata_hash_15mer.csv"
    SAVE_DIR = r"E:\pycharm\pycharm_library\neural_network\LSHLSTM\data\RawData"

    loader = create_windows_loader(CSV_PATH, EMB_PATH, 256)


    batch_num = 0
    for batch in tqdm(loader, desc="Saving batches"):
        features, labels = batch


        np.save(os.path.join(SAVE_DIR, f'batch_{batch_num}_features.npy'), features.numpy())
        np.save(os.path.join(SAVE_DIR, f'batch_{batch_num}_labels.npy'), labels.numpy())

        del batch
        batch_num += 1