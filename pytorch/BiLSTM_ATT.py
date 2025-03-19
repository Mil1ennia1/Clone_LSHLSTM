import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau



class TrainingConfig:
    def __init__(self):
        # 数据配置
        self.data_dir = r"E:\pycharm\pycharm_library\neural_network\LSHLSTM\data\train_data\cut_train_noisy_15mer"  # 根目录
        self.feature_dim = 100
        self.sequence_length = 136

        # 模型配置
        self.hidden_dim = 256
        self.lstm_layers = 2
        self.attention_heads = 4
        self.num_classes = 2
        self.dropout = 0.3

        # 训练配置
        self.batch_size = 64
        self.learning_rate = 0.001
        self.epochs = 100
        self.test_size = 0.2
        self.random_seed = 42


class GenomeSequenceDataset(Dataset):
    def __init__(self, feature_paths, label_paths, config):
        self.config = config
        self.feature_paths = feature_paths
        self.label_paths = label_paths
        self._validate_files()

    def _validate_files(self):
        assert len(self.feature_paths) == len(self.label_paths), \
            "特征文件和标签文件数量不匹配"
        for f, l in zip(self.feature_paths, self.label_paths):
            f_id = os.path.basename(f).split("_")[0]
            l_id = os.path.basename(l).split("_")[0]
            assert f_id == l_id, f"文件不匹配: {f} vs {l}"

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        features = np.load(self.feature_paths[idx])
        label = np.load(self.label_paths[idx])[0]
        label = int(label)

        seq_len = features.shape[0]
        actual_length = min(seq_len, self.config.sequence_length)
        padded_features = np.zeros((self.config.sequence_length, self.config.feature_dim))
        padded_features[:actual_length] = features[:actual_length]

        return (
            torch.from_numpy(padded_features).float(),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(actual_length, dtype=torch.long)
        )


class BiLSTMAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.feature_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.lstm_layers,
            bidirectional=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
            batch_first=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim * 2,
            num_heads=config.attention_heads,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.num_classes)
        )

    def forward(self, x, lengths):


        lengths = lengths.cpu()

        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        lstm_out, _ = self.lstm(packed_x)

        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        attn_output, _ = self.attention(
            lstm_out, lstm_out, lstm_out
        )  # (batch_size, sequence_length, hidden_dim*2)

        last_outputs = []
        for i, length in enumerate(lengths):
            last_outputs.append(attn_output[i, length - 1, :])
        last_outputs = torch.stack(last_outputs)

        return self.classifier(last_outputs)

class TrainingSystem:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics_file = "training_metrics.txt"


        feature_paths = sorted(glob.glob(os.path.join(config.data_dir, "batch_*_features.npy")))
        label_paths = sorted(glob.glob(os.path.join(config.data_dir, "batch_*_labels.npy")))

        train_feature, val_feature, train_label, val_label = train_test_split(
            feature_paths, label_paths,
            test_size=config.test_size,
            random_state=config.random_seed
        )

        self.train_set = GenomeSequenceDataset(train_feature, train_label, config)
        self.val_set = GenomeSequenceDataset(val_feature, val_label, config)

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=config.batch_size,
            num_workers=4
        )

        self.model = BiLSTMAttention(config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        with open(self.metrics_file, 'w') as f:
            f.write("epoch,train_loss,val_loss,accuracy,precision,recall,f1,time,current_lr\n")

    def _record_metrics(self, epoch, train_loss, val_loss, acc, precision, recall, f1, epoch_time, current_lr):

        with open(self.metrics_file, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{acc:.4f},")
            f.write(f"{precision:.4f},{recall:.4f},{f1:.4f},{epoch_time:.1f},{current_lr:.8f}\n")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        for features, labels, lengths in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
            features = features.to(self.device)  # (B, sequence_length, feature_dim)
            labels = labels.to(self.device)  # (B,)
            lengths = lengths.to(self.device)  # (B,)

            self.optimizer.zero_grad()

            outputs = self.model(features, lengths)  # (B, num_classes)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels, lengths in self.val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)

                outputs = self.model(features, lengths)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        return total_loss / len(self.val_loader), acc, precision, recall, f1

    def run(self):
        best_f1 = 0.0

        for epoch in range(1, self.config.epochs + 1):
            start_time = time.time()

            # 训练阶段
            train_loss = self.train_epoch(epoch)

            # 验证阶段
            val_loss, acc, precision, recall, f1 = self.validate()

            epoch_time = time.time() - start_time

            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录指标
            self._record_metrics(
                epoch, train_loss, val_loss, acc,
                precision, recall, f1, epoch_time, current_lr
            )

            # 打印结果
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Accuracy: {acc:.2%} | Precision: {precision:.2%}")
            print(f"Recall: {recall:.2%} | F1: {f1:.2%}")
            print(f"Time: {epoch_time:.1f}s")
            print(f"Current Learning Rate: {current_lr:.8f}")

            # 调整学习率
            self.scheduler.step(val_loss)

            # 保存最佳模型
            if f1 > best_f1:
                best_f1 = f1
                torch.save(self.model.state_dict(), "best_model.pth")

        # 保存最终模型
        torch.save(self.model.state_dict(), "final_model.pth")

if __name__ == "__main__":
    config = TrainingConfig()
    trainer = TrainingSystem(config)
    trainer.run()