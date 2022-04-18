from numbers import Number
from typing import Sequence
import dask.dataframe as dd

def check_size_factor(size: int=None, factor=None):
    if size is None != factor is None:
        raise ValueError("Either enter the size or the factor")
    
    if (size is not None and size <= 0) or (factor is not None and factor <= 0):
        raise ValueError("Sample needs to be a positive number")

def sample(inpath, outpath, size: int=None, factor=None, balance=False, random_state=None):
    check_size_factor(size, factor)
    
    if not balance:
        sample_unbalanced(inpath, outpath, size, factor, random_state)
    else:
        sample_balanced(inpath, outpath, size, factor, random_state)

def sample_balanced(inpath, outpath, size: int=None, factor=None, random_state=None):
    check_size_factor(size, factor)
    ddf: dd.DataFrame = dd.read_csv(inpath)
    
    if size is None:
        size = int(factor*len(ddf))

    pos = ddf.loc[ddf['polarity'] == 1]
    neu = ddf.loc[ddf['polarity'] == 0]
    neg = ddf.loc[ddf['polarity'] == -1]

    frac_pos = size/len(pos) + 1e-5
    frac_neu = size/len(neu) + 1e-5
    frac_neg = size/len(neg) + 1e-5

    sample_pos = pos.sample(frac=frac_pos, replace=frac_pos>1, random_state=random_state).head(size, -1)
    sample_neu = neu.sample(frac=frac_neu, replace=frac_neu>1, random_state=random_state).head(size, -1)
    sample_neg = neg.sample(frac=frac_neg, replace=frac_neg>1, random_state=random_state).head(size, -1)

    sample = dd.concat([sample_pos, sample_neu, sample_neg]).sample(frac=1, random_state=random_state)

    sample.to_csv(outpath)

def sample_unbalanced(inpath, outpath, size: int=None, factor=None, random_state=None):
    check_size_factor(size, factor)
    ddf: dd.DataFrame = dd.read_csv(inpath)
    
    if size is not None:
        frac = size/len(ddf) + 1e-5
        sample = ddf.sample(frac=frac, replace=frac>1, random_state=random_state).head(size, -1, False)
    elif isinstance(factor, int):
        sample = dd.concat([ddf] * factor).sample(frac=1, random_state=random_state)
    else:
        sample = ddf.sample(frac=factor, replace=factor>1, random_state=random_state)

    sample.to_csv(outpath)

def split_csv(inpath: str, outpaths: Sequence[str], frac: Sequence[Number], random_state=None, shuffle=False):
    if len(frac) != len(outpaths):
        raise ValueError(f"The number of outputs does not match the number of output files")
    
    ddf: dd.DataFrame = dd.read_csv(inpath, dtype={'polarity': 'object'}, engine="python", on_bad_lines='skip')
    populations = ddf.random_split(frac, random_state, shuffle)

    for outpath, population in zip(outpaths, populations):
        population.to_csv(outpath)

if __name__ == "__main__":
    yelp_pop_train_path = 'ExternalData/yelp/population/train-*.csv'
    yelp_pop_test_path = 'ExternalData/yelp/population/test-*.csv'

    yelp_pret_train_path = 'ExternalData/yelp/pret/train-*.csv'
    yelp_pret_test_path = 'ExternalData/yelp/pret/test-*.csv'

    yelp_mult_train_path = 'ExternalData/yelp/2015/mult/train-*.csv'
    yelp_mult_test_path = 'ExternalData/yelp/2015/mult/test-*.csv'

    sem_rest_mult_train_path = 'ExternalData/semeval_2015/restaurant/mult/train-*.csv'
    sem_rest_mult_test_path = 'ExternalData/semeval_2015/restaurant/mult/test-*.csv'

    sem_rest_ft_train_path = 'ExternalData/semeval_2015/restaurant/ft/train.csv'
    sem_rest_ft_test_path = 'ExternalData/semeval_2015/restaurant/ft/test.csv'

    # split large file into training set and test set
    # print('splitting large file')
    # split_csv('ExternalData/yelp.csv', [yelp_pop_train_path, yelp_pop_test_path],
    # [0.8, 0.2])

    # Sample for pretraining
    # print('sampling pret documents')
    # sample(yelp_pop_train_path, yelp_pret_train_path, size=10_000, balance=True)
    # sample(yelp_pop_test_path, yelp_pret_test_path, size=1_000)

    # Sample aspect for mult
    print('sampling mult aspects')
    sample(sem_rest_ft_train_path, sem_rest_mult_train_path, factor=3)
    sample(sem_rest_ft_test_path, sem_rest_mult_test_path, factor=1)

    # Sample documents for mult
    # unbalanced? or should we balance?
    print('sampling mult documents')
    size_asp_train = len(dd.read_csv(sem_rest_mult_train_path))
    size_asp_test = len(dd.read_csv(sem_rest_mult_test_path))

    sample(yelp_pop_train_path, yelp_mult_train_path, size=size_asp_train)
    sample(yelp_pop_test_path, yelp_mult_test_path, size=size_asp_test)
