'''使用TSNE进行数据可视化'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
seqss = pd.read_csv('/home/chen/chengxu/LSHVec-master/my/seqs.csv')
def sample_data(data, num_samples):
    samples = []
    for label in data['special'].unique()[:10]:
        # get rows with the current label
        df = data[data['special'] == label]
        # randomly select num_samples rows
        sampled_df = df.sample(n=num_samples, random_state=42)
        samples.append(sampled_df)
    # concatenate the results and return
    return pd.concat(samples)
seqs_sampled = sample_data(seqss, 1000)
s1 = seqs_sampled['vec']
seqs_sampled['vec_list'] = s1.apply(lambda x: [float(i) for i in x.strip('[]').split()])
X=np.array(list(seqs_sampled['vec_list'].values))
tsne = TSNE(n_components=2)
Y = tsne.fit_transform(X)
labels = seqs_sampled['special'].values
legends = list(set(labels))
for label in legends:
    plt.scatter(Y[labels==label][:,0], Y[labels==label][:,1], alpha=0.5, s=1)
plt.legend(legends)
plt.savefig('/home/chen/chengxu/LSHVec-master/my/train5.png')
plt.show()

'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 读取数据集
seqss = pd.read_csv('/home/chen/chengxu/LSHVec-master/my/seqs.csv')

# 抽取每个类别的 50 个样本
samples = []
for label in seqss['special'].unique()[:10]:
    df = seqss[seqss['special'] == label]
    sampled_df = df.sample(n=500, random_state=42)
    samples.append(sampled_df)
seqs_sampled = pd.concat(samples)

# 将向量解析为浮点数列表
s1 = seqs_sampled['vec']
seqs_sampled['vec_list'] = s1.apply(lambda x: [float(i) for i in x.strip('[]').split()])

# 构造输入矩阵 X
X = np.array(list(seqs_sampled['vec_list'].values))

# 使用 PCA 进行降维
pca = PCA(n_components=2)
Y = pca.fit_transform(X)

# 绘制散点图
labels = seqs_sampled['special'].values
legends = list(set(labels))
for label in legends:
    plt.scatter(Y[labels==label][:,0], Y[labels==label][:,1], alpha=0.5, s=1)
plt.savefig('/home/chen/chengxu/LSHVec-master/my/train2.png')
plt.show()'''