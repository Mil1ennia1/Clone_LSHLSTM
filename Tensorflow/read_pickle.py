import pandas as pd
import tensorflow as tf
# 读取pickle文件
seqs = pd.read_pickle('/home/chen/chengxu/LSHVec-master/my/test1out.pickle')

# 读取'vec'列的数据
vecs = seqs['id'][1]
vecss = seqs['id'][2]
list=[]
list.append(vecs)
list.append(vecss)
#print(type(list))

'''graph = tf.Graph()
with graph.as_default():
    # 创建Tensor并将操作添加到计算图
    my_tensor1 = tf.constant(list)

with tf.compat.v1.Session(graph=graph) as sess:
    # 使用Session计算张量的值
    batch_features = sess.run(my_tensor1)
    print(batch_features.shape)
# 输出结果'''
print(vecs)
print(type(vecs))
print(vecss)
print(type(vecss))
