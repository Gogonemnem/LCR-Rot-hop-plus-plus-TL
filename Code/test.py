import dask.dataframe as dd

def sample(df: dd.DataFrame, size: int=None, balance: bool=True):
    if balance:
        pos: dd.DataFrame = df.loc[df['polarity'] == '1']
        frac = 1 if size is None else size/len(pos) + 1e-5 # prevent rounding errors
        pos = pos.sample(frac=frac, replace=frac>1).head(size, -1)
        print('total positives', len(pos))

        neu: dd.DataFrame = df.loc[df['polarity'] == '0']
        frac = 1 if size is None else size/len(neu) + 1e-5
        neu = neu.sample(frac=frac, replace=frac>1).head(size, -1)
        print('total neutral', len(neu))

        neg: dd.DataFrame = df.loc[df['polarity'] == '-1']
        frac = 1 if size is None else size/len(neg) + 1e-5
        neg = neg.sample(frac=frac, replace=frac>1).head(size, -1)
        print('total negatives', len(neg))

        sample = dd.concat([pos, neu, neg]).sample(frac=1)
    else:
        frac = 1 if size is None else size/len(df) + 1e-5
        sample = df.sample(frac=frac, replace=frac>1)
        
    print('total sample', len(sample))
    return sample
        

# df: dd.DataFrame = dd.read_csv('ExternalData/yelp.csv', on_bad_lines='skip', engine="python", dtype={'polarity': 'object'})
# size = 10_000
# pos: dd.DataFrame = df.loc[df['polarity'] == '1']
# pos = pos.sample(frac=(size+10)/len(pos), replace=(size+10)/len(pos)>1).head(size, -1)
# print(len(pos))
# neu: dd.DataFrame = df.loc[df['polarity'] == '0']
# neu = neu.sample(frac=(size+10)/len(neu), replace=(size+10)/len(neu)>1).head(size, -1)
# print(len(neu))
# neg: dd.DataFrame = df.loc[df['polarity'] == '-1']
# neg = neg.sample(frac=(size+10)/len(neg), replace=(size+10)/len(neg)>1).head(size, -1)
# print(len(neg))

# sample = dd.concat([pos, neu, neg]).sample(frac=1)
# print(len(sample))

# sample.to_csv('ExternalData/yelp/yelp-*.csv')  

# df: dd.DataFrame = dd.read_csv('ExternalData/yelp/*.csv')
# print(df.head(5))
# print(len(df))
# print(df.compute().T.values)