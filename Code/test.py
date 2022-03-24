import dask.dataframe as dd

df: dd.DataFrame = dd.read_csv('ExternalData/yelp.csv', on_bad_lines='skip', engine="python", dtype={'polarity': 'object'})
# print(df.info())
# for i in range(10):
#     print(df.sample(frac=.1).head(5))
# df['polarity'] = df['polarity'].astype(int)
size = 10_000
pos: dd.DataFrame = df.loc[df['polarity'] == '1']
pos = pos.sample(frac=size/len(pos), replace=size/len(pos)>1)
neu: dd.DataFrame = df.loc[df['polarity'] == '0']
neu = neu.sample(frac=size/len(neu), replace=size/len(neu)>1)
neg: dd.DataFrame = df.loc[df['polarity'] == '-1']
neg = neg.sample(frac=size/len(neg), replace=size/len(neg)>1)

print(len(pos), len(neu), len(neg))

