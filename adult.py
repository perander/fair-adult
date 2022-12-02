#%%
import pandas as pd
import numpy as np
#%%
def preprocess(df, categorical_variables):
    for col in df.columns:
        if df[col].isnull().values.any():
            df = df[col].fillna(df[col].mode()[0])

    orig_df = df

    for col in categorical_variables:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)

    df = df.drop(labels=categorical_variables, axis=1)

    return df, orig_df

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

    adult, orig_adult = preprocess(adult, categorical_variables)

    adult_train = adult.iloc[:adult_train_og_dim[0], :]
    adult_test = adult.iloc[adult_train_og_dim[0]:, :]

    adult_train = adult_train.replace([' <=50K', ' >50K'],[1,0])
    adult_test = adult_test.replace([' <=50K.', ' >50K.'],[1,0])

    orig_adult_train = orig_adult.iloc[:adult_train_og_dim[0], :]
    orig_adult_test = orig_adult.iloc[adult_train_og_dim[0]:, :]

    orig_adult_train = orig_adult_train.replace([' <=50K', ' >50K'],[1,0])
    orig_adult_test = orig_adult_test.replace([' <=50K.', ' >50K.'],[1,0])

    return adult_train, adult_test, orig_adult_train, orig_adult_test
#%%
adult_train, adult_test, orig_adult_train, orig_adult_test = load_adult()
print(adult_train.shape)
print(orig_adult_train.shape)
print(orig_adult_test.shape)

#%%
train_X = adult_train.drop('income', axis=1)
train_Y = adult_train['income']
test_X = adult_test.drop('income', axis=1)
test_Y = adult_test['income']

print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)

train_X = np.array(train_X, dtype=np.float32)
test_X = np.array(test_X, dtype=np.float32)
train_Y = np.array(train_Y, dtype=np.int32)
test_Y = np.array(test_Y, dtype=np.int32)
#%%

from sklearn.ensemble import RandomForestClassifier

for i in range(19,20):
    model = RandomForestClassifier(max_depth=i, random_state=0)
    model.fit(train_X, train_Y)
    pred = model.predict(test_X)
    print(i, model.score(test_X, test_Y))

#%%
print(adult_train.columns)

#%%
print(pred.shape)
print(test_X.shape)
print(test_Y.shape)
pred = pred[:, np.newaxis]
test_Y = test_Y[:,np.newaxis]

#%%
orig_prediction = np.concatenate((pred, test_Y), axis=1)
orig_prediction = np.concatenate((orig_prediction, orig_adult_test.drop("income", axis=1)), axis=1)

print(orig_prediction.shape)
#%%
cols = orig_adult_test.drop("income", axis=1).columns
cols = np.concatenate((np.array(["score", "label_value"]), cols))
print(cols)
#%%
df = pd.DataFrame(orig_prediction, columns=cols)
print(df)

#%%
df.to_csv("predictions.csv", index=False)
