'''s="label|2|.11861_6_61.1-9598/1"
lst=s.split('|')
print(lst)
print(s.split('|')[1])'''
'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE'''

'''def read_embedding(fname):
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
with open("/home/chen/chengxu/LSHVec-master/my/data15.fnv") as fin:
    lines = list(fin)
for i,line in enumerate(lines):
    words = [int(u) for u in line.strip().split(" ")]
    vec = [embedding[u] for u in words if u in embedding]
    seqvectors.append(np.mean(vec,0) if len(vec)>0 else None)
seqs = pd.read_csv("/home/chen/chengxu/LSHVec-master/my/data.seq",sep='\t', header=None, usecols=[0,1],index_col=0)
seqs.columns=['id']
seqs['special'] = seqs['id'].apply(lambda x: x.split("|")[1])
seqs['vec']=seqvectors
seqs.to_csv('/home/chen/chengxu/LSHVec-master/my/seqs.csv', sep=',', index=False)'''
'''seqss=pd.read_csv("/home/chen/chengxu/LSHVec-master/my/seqs.csv")
tsne = TSNE(n_components=2, perplexity=100, n_jobs=8)
X=np.array(list(seqss['vec'].values))
X.shape
Y = tsne.fit_transform(X)
labels=seqss['special'].values
legends=list(set(labels))
for label in legends:
    plt.scatter(Y[labels==label][:,0],Y[labels==label][:,1],alpha=0.5,s=1)
plt.legend(legends)
plt.savefig('/home/chen/chengxu/LSHVec-master/my/train.png')
plt.show()'''
'''import pandas as pd
seqs=[[1,2],[2,2],[1,1],[3,2],[4,2],[5,2]]
s=pd.DataFrame.from_dict(seqs)
s.head()
print(s.head())'''
'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
seqss = pd.read_csv('/home/chen/chengxu/LSHVec-master/my/seqs.csv')
def sample_data(data, num_samples):
    samples = []
    for label in data['special'].unique():
        # get rows with the current label
        df = data[data['special'] == label]
        # randomly select num_samples rows
        sampled_df = df.sample(n=num_samples, random_state=42)
        samples.append(sampled_df)
    # concatenate the results and return
    return pd.concat(samples)
seqs_sampled = sample_data(seqss, 1000)
s=seqs_sampled['vec'].values
print(s[0],type(s[0]))'''
'''X=np.array(list(seqs_sampled['vec'].values))
X = X.reshape((X.shape[0], -1))
tsne = TSNE(n_components=2, perplexity=100, n_jobs=8)
Y = tsne.fit_transform(X)


labels = seqs_sampled['special'].values
legends = list(set(labels))

for label in legends:
    plt.scatter(Y[labels==label][:,0], Y[labels==label][:,1], alpha=0.5, s=1)

plt.savefig('/home/chen/chengxu/LSHVec-master/my/train.png')
plt.show()'''
import pandas as pd
import numpy as np
# 读取 CSV 文件
df = pd.read_csv('/home/chen/chengxu/LSHVec-master/my/test.csv')

# 获取包含浮点数字符串的 Series 对象
s1 = df['vec']

# 将每个字符串转为浮点数列表，并将结果作为新列添加到 DataFrame 中
df['vec_list'] = s1.apply(lambda x: [float(i) for i in x.strip('[]').split()])

# 打印结果
print(df['vec_list'])
X=np.array(list(df['vec_list'].values))
print(X,X.shape)
