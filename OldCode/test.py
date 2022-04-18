import dask.dataframe as dd

def sample_old(df: dd.DataFrame, size: int=None, balance: bool=True):
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

def compute_fraction_size(df: dd.DataFrame, size=None):
    if size is None:
        return 1

    if size == 0:
        return size

    return (size)/len(df) + 1e-5

def sample_unbalanced(df: dd.DataFrame, size=None):
    frac = compute_fraction_size(df, size)
    sample = df.sample(frac=frac, replace=frac>1)
    return sample

def sample_balanced(df: dd.DataFrame, size=None):
    pos = df.loc[df['polarity'] == '1']
    train_pos = sample(pos, size)

    neu = df.loc[df['polarity'] == '0']
    train_neu = sample(neu, size)

    neg = df.loc[df['polarity'] == '-1']
    train_neg = sample(neg, size)

    train = dd.concat([train_pos, train_neu, train_neg]).sample(frac=1)
    return train

def sample(df: dd.DataFrame, balance=False, size=None, validation_size=0, frac=[0.8, 0.2]):
    if validation_size:
        train_pop, valid_pop = df.random_split(frac)
    else:
        train_pop = df

    sample_function = sample_balanced if balance else sample_unbalanced

    train = sample_function(train_pop, size)
    if validation_size:
        valid = sample_unbalanced(valid_pop, validation_size)
        return train, valid

    return train

def sample_csv(inpath, out_train_path=None, out_valid_path=None,  size=None, validation_size=0, balance: bool=False):
    df: dd.DataFrame = dd.read_csv(inpath, dtype={'polarity': 'object'}, engine="python", on_bad_lines='skip')

    if validation_size:
        train, valid = sample(df, balance, size, validation_size)
        
        if out_valid_path:
            valid.to_csv(out_valid_path)
    else:
        train = sample(df, balance, size, validation_size)
    
    if out_train_path:
        train.to_csv(out_train_path)

if __name__ == "__main__":
    sample_csv('ExternalData/yelp.csv', out_train_path='ExternalData/_yelp/train-*.csv', out_valid_path='ExternalData/_yelp/valid-*.csv',  size=3000, validation_size=1000, balance=False)
