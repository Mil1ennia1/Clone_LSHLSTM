import pandas as pd
import tensorflow as tf
import numpy as np
def input_fn_train():
    # Load embedding vectors and labels from CSV file using pandas
    df = pd.read_csv("/home/chen/chengxu/LSHVec-master/my/encodiing.csv")
    column_data = df["kmer"].apply(lambda x: [int(i) for i in x.strip('[]').split(',')])  # 将某一列中的字符串数据转化列表
    data = column_data.to_list()
    my_features = tf.constant(data)
    column_data1= df["special"]
    data1 = column_data1.to_list()
    my_label = tf.constant(data1)
    features = my_features  # drop the label column
    labels = my_label
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    # Shuffle, repeat, and batch the dataset
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat()
    dataset = dataset.batch(2)
    # Create an iterator for the dataset and return the next batch of data
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels
def read_embedding(fname):
    with open(fname) as fin:
        lines=list(fin)
    assert len(lines)>1
    first_line=lines[0]
    num_word, dim_vec = [int(u) for u in first_line.split(" ")]
    assert len(lines)==num_word+1
    ret ={}
    for line in lines[1:]:
        lst = line.strip().split(" ")
        assert len(lst)==dim_vec+1, line
        if lst[0]=='</s>': continue
        word = int(lst[0])
        vec = np.array([float(u) for u in lst[1:]])
        ret[word]=vec
    return ret,dim_vec
embedding,dim_vec = read_embedding("/home/chen/chengxu/LSHVec-master/my/lshvec.vec")
features, labels=input_fn_train()
seqvectors=[]
for line in features:
    vec = [embedding[u] for u in line if u in embedding]
    seqvectors.append(vec if len(vec) > 0 else None)
print(np.array(seqvectors))