#将序列的编码保存在tfrecord文件中
import pandas as pd
import numpy as np
import tensorflow as tf
seqvectors = []
with open("/home/chen/chengxu/LSHVec-master/my/test_input/val_fatsa.seq") as fin: #编码的文件
    lines = list(fin)
for i,line in enumerate(lines):
    words = [int(u) for u in line.strip().split(" ")]
    seqvectors.append(words)
seqs = pd.read_csv("/home/chen/chengxu/LSHVec-master/my/test_input/val_fasta.csv",sep='\t', header=None, usecols=[0,1],index_col=0)
seqs.columns=['id']
seqs['special'] = seqs['id'].apply(lambda x: x.split("|")[1])
seqs['kmer']=seqvectors
#seqs.to_csv('/home/chen/chengxu/LSHVec-master/my/encodiing.csv', sep=',', index=False)
with tf.io.TFRecordWriter("/home/chen/chengxu/LSHVec-master/my/test_input/val_test.tfrecord") as writer:
    for index, row in seqs.iterrows():
        #label_id = row['special']
        kmer_array = row['kmer']
        data = \
            {
                'read': tf.train.Feature(int64_list=tf.train.Int64List(value=kmer_array)),
                #'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label_id)]))
            }
        feature = tf.train.Features(feature=data)
        example = tf.train.Example(features=feature)
        serialized = example.SerializeToString()
        writer.write(serialized)