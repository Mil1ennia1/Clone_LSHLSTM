'''将通过skipgram得到的k-mer嵌入向量，矢量化整个序列（所有的kmer求平均值），并保存在一个csv文件中'''
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
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
seqvectors = []
with open("/home/chen/chengxu/LSHVec-master/my/test1.csv") as fin: #编码的文件
    lines = list(fin)
for i,line in enumerate(lines):
    words = [int(u) for u in line.strip().split(" ")]
    vec = [embedding[u] for u in words if u in embedding]
    #seqvectors.append(np.mean(vec,0) if len(vec)>0 else None)
    seqvectors.append(vec if len(vec) > 0 else None)
seqs = pd.read_csv("/home/chen/chengxu/LSHVec-master/my/test1.seq",sep='\t', header=None, usecols=[0,1],index_col=0)
seqs.columns=['id']
seqs['special'] = seqs['id'].apply(lambda x: x.split("|")[1])
seqs['vec']=seqvectors
#dtypes = {'special': int,'id':str,'vec':bytearray}
#seqs = seqs.astype(dtypes)
#seqs = seqs.set_index('id')
#seqs.to_csv('/home/chen/chengxu/LSHVec-master/my/test2.csv', sep=',', index=False)
#seqs.to_pickle('/home/chen/chengxu/LSHVec-master/my/data15.pickle') #保存的位置
'''def create_example(row):
    """
    将每行数据转换为一个Example对象
    """
    # 从行中获取数据
    id = row['id']
    special = row['special']
    vec = row['vec']

    # 创建Feature对象
    feature = {
        'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[id.encode()])),
        'special': tf.train.Feature(bytes_list=tf.train.BytesList(value=[special.encode()])),
        'vec': tf.train.Feature(float_list=tf.train.FloatList(value=vec)),
    }

    # 创建Example对象
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # 序列化Example对象
    serialized = example.SerializeToString()

    return serialized

# 将数据保存到TFRecord文件
with tf.io.TFRecordWriter('/home/chen/chengxu/LSHVec-master/my/test2.tfrecord') as writer:
    for index, row in seqs.iterrows():
        serialized = create_example(row)
        writer.write(serialized)'''

