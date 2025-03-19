import pandas as pd
seqs = pd.read_csv("/home/chen/chengxu/LSHVec-master/my/test1.seq",sep='\t', header=None, usecols=[0,1],index_col=0)
seqs.columns=['id']
seqs['special'] = seqs['id'].apply(lambda x: x.split("|")[1])
print(type(seqs['special'][1]))
'''import csv
# 打开csv文件
with open('/home/chen/chengxu/LSHVec-master/my/out/label.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # 写入表头
    writer.writerow(['special', 'label'])
    # 遍历unique的label并写入csv文件
    for idx, label in enumerate(seqs['special'].unique()):
        writer.writerow([label, idx])'''
# 读取output.csv文件
output_df = pd.read_csv('/home/chen/chengxu/LSHVec-master/my/out/label.csv')

# 创建一个字典，将label映射为对应的id值
label_to_id = {str(id): label for id, label in output_df[['special', 'label']].values}
print(label_to_id['2'],type(label_to_id['2']))
# 将seqs['special']中的值替换为对应的id值
seqs['label'] = seqs['special'].map(label_to_id)
print(seqs['label'])
