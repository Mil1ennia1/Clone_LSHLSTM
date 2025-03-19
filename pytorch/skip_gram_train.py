import csv
from gensim.models import Word2Vec
import os
import tempfile
import matplotlib.pyplot as plt
from tqdm import tqdm

class Config:
    input_path = r"E:\pycharm\pycharm_library\neural_network\LSHLSTM\data\RawData\cutdata_hash_15mer.csv"
    output_model = r"E:\pycharm\pycharm_library\neural_network\LSHLSTM\LSHVEC\cutdata_15mer_model.pth"
    vector_size = 100
    window = 2
    min_count = 1
    workers = os.cpu_count() - 1
    epochs = 20
    sg = 1
    hs = 0
    negative = 3
    sample = 1e-5

def create_corpus_file():
    temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
    with open(Config.input_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in tqdm(reader, desc="Processing CSV rows", unit="rows"):
            if len(row) >= 3:
                hash_values = row[2].split()
                temp_file.write(' '.join(hash_values) + '\n')
    temp_file.close()
    return temp_file.name

def train_model(corpus_file):
    sentences = []
    with open(corpus_file, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Reading corpus file", unit="lines"):
            sentence = line.strip().split()
            sentences.append(sentence)

    model = Word2Vec(
        sentences=sentences,
        vector_size=Config.vector_size,
        window=Config.window,
        min_count=Config.min_count,
        workers=Config.workers,
        sg=Config.sg,
        hs=Config.hs,
        negative=Config.negative,
        sample=Config.sample,
        epochs=Config.epochs,
        compute_loss=True
    )

    losses = []
    initial_epoch_loss = model.get_latest_training_loss()

    for epoch in tqdm(range(Config.epochs), desc="Training epochs", unit="epoch"):
        model.train(sentences, total_examples=model.corpus_count, epochs=1)
        current_epoch_loss = model.get_latest_training_loss()
        epoch_loss = current_epoch_loss - initial_epoch_loss
        losses.append(epoch_loss)
        initial_epoch_loss = current_epoch_loss

    return model, losses

# 主程序
if __name__ == "__main__":
    corpus_file = create_corpus_file()

    model, losses = train_model(corpus_file)

    os.remove(corpus_file)

    model.save(Config.output_model)
    model.wv.save_word2vec_format("15mer_vectors.vec", binary=False)

    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    test_kmer = "10000000" 
    if test_kmer in model.wv:
        print(f"\n向量维度: {model.wv[test_kmer].shape}")
        print(f"\n相似k-mer (前5个):")
        for word, sim in model.wv.most_similar(test_kmer, topn=5):
            print(f"{word}: {sim:.4f}")
    else:
        print(f"k-mer {test_kmer} 未被收录（低于min_count阈值）")