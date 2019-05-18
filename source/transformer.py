from sklearn.base import BaseEstimator, TransformerMixin


def clean_nulls(df, cols):
    for col in cols:
        df[col].fillna((df[col].mean()), inplace=True)


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class SamplerFactory:
    __sampler = {
        'over': SMOTE,
        'under': RandomUnderSampler
    }

    @staticmethod
    def get_instance(sample_method, *args, **kwargs):
        sampler = SamplerFactory.__sampler.get(sample_method)
        if sampler is not None:
            return sampler(*args, **kwargs)
        else:
            raise ValueError('invalid parameter: {}'.format(sample_method))
