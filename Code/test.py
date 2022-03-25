import dask.dataframe as dd

df: dd.DataFrame = dd.read_csv('ExternalData/yelp.csv', on_bad_lines='skip', engine="python", dtype={'polarity': 'object'})
# print(df.info())
# for i in range(10):
#     print(df.sample(frac=.1).head(5))
# df['polarity'] = df['polarity'].astype(int)
size = 10_000
pos: dd.DataFrame = df.loc[df['polarity'] == '1']
pos = pos.sample(frac=(size+10)/len(pos), replace=(size+10)/len(pos)>1).head(size, -1)
print(len(pos))
neu: dd.DataFrame = df.loc[df['polarity'] == '0']
neu = neu.sample(frac=(size+10)/len(neu), replace=(size+10)/len(neu)>1).head(size, -1)
print(len(neu))
neg: dd.DataFrame = df.loc[df['polarity'] == '-1']
neg = neg.sample(frac=(size+10)/len(neg), replace=(size+10)/len(neg)>1).head(size, -1)
print(len(neg))

# neu: dd.DataFrame = df.loc[df['polarity'] == '0']
# neu = neu.sample(frac=(size+10)/len(neu), replace=(size+10)/len(neu)>1).reset_index().loc['0': f'{size}']
# neg: dd.DataFrame = df.loc[df['polarity'] == '-1']
# neg = neg.sample(frac=(size+10)/len(neg), replace=(size+10)/len(neg)>1).reset_index().loc['0': f'{size}']

sample = dd.concat([pos, neu, neg]).sample(frac=1)
print(len(sample))

sample.to_csv('ExternalData/yelp/yelp-*.csv')  

