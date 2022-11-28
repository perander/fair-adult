#%%
import pandas as pd

#%%
def preprocess(df, categorical_variables):
    for col in df.columns:
        if df[col].isnull().values.any():
            df = df[col].fillna(df[col].mode()[0])

    for col in categorical_variables:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)

    df = df.drop(labels=categorical_variables, axis=1)

    return df

#%%
def load_adult():
    train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

    adult_train = pd.read_csv(train, skiprows=0, index_col=False, names=columns)
    adult_test = pd.read_csv(test, skiprows=1, index_col=False, names=columns)

    adult_train_og_dim = adult_train.shape
    adult = pd.concat([adult_train, adult_test], axis=0)

    categorical_variables = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

    adult = preprocess(adult, categorical_variables)

    adult_train = adult.iloc[:adult_train_og_dim[0], :]
    adult_test = adult.iloc[adult_train_og_dim[0]:, :]

    adult_train = adult_train.replace([' <=50K', ' >50K'],[1,0])
    adult_test = adult_test.replace([' <=50K.', ' >50K.'],[1,0])

    return adult_train, adult_test
#%%
adult_train, adult_test = load_adult()
print(adult_train.shape)
