import pandas as pd
import numpy as np
import tensorflow as tf

seqs = pd.read_csv("/home/chen/chengxu/LSHVec-master/my/test1.seq",sep='\t', header=None, usecols=[0,1],index_col=0)
seqs.columns=['id']
seqs['special'] = seqs['id'].apply(lambda x: x.split("|")[1])
output_df = pd.read_csv('/home/chen/chengxu/LSHVec-master/my/out/label.csv')
label_to_id = {str(id): label for id, label in output_df[['special', 'label']].values}
seqs['label'] = seqs['special'].map(label_to_id)
i = 1
#print(seqs['label'])
with tf.io.TFRecordWriter("/home/chen/chengxu/LSHVec-master/my/test_val.tfrecord") as writer:
    with open("/home/chen/chengxu/LSHVec-master/my/test1.csv") as fin: #/home/chen/chengxu/LSHVec-master/my/data15.fnv
        for line in fin:
            words = [int(u) for u in line.strip().split(" ")]
            label_id = seqs['label'][i]
            i=i+1
            kmer_array = words
            data = \
                {
                    'read': tf.train.Feature(int64_list=tf.train.Int64List(value=kmer_array)),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label_id)]))
                }
            feature = tf.train.Features(feature=data)
            example = tf.train.Example(features=feature)
            serialized = example.SerializeToString()
            writer.write(serialized)




