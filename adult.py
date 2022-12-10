#%%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
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

    adult_train = adult_train.replace([' <=50K', ' >50K'],[0,1])
    adult_test = adult_test.replace([' <=50K.', ' >50K.'],[0,1])

    orig_adult_train = orig_adult.iloc[:adult_train_og_dim[0], :]
    orig_adult_test = orig_adult.iloc[adult_train_og_dim[0]:, :]

    orig_adult_train = orig_adult_train.replace([' <=50K', ' >50K'],[0,1])
    orig_adult_test = orig_adult_test.replace([' <=50K.', ' >50K.'],[0,1])

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
# fairness correction: preferential resampling

def preferential_resampling(dataset, s_names, x_name, y_name='income'):
    """ For each group s with the same value of x, replace rows so that p(y=1) is as similar as possible for all groups s."""

    values = dataset[x_name].unique()
    
    for i, value in enumerate(sorted(values)):
        print(f"{i+1}/{len(values)}, value", value)
        dataset_with_value = dataset[dataset[x_name] == value]

        # get protected groups (e.g. get groups of males and females)
        groups = []
        lens = []
        for s_name in s_names:
            group = dataset_with_value[dataset_with_value[s_name] == 1]
            groups.append(group)
            lens.append(group.shape[0])
        
        # if all but one group are empty, skip preprocessing for that value
        if lens.count(0) >= len(groups)-1:
            print("did not do any preprocessing")
            continue
        
        # calculate the ratio of >50K (1) and <=50K (0) for each group (e.g. males and females)
        ratios = []
        for group in groups:
            n_group = group.shape[0]

            if n_group > 0:
                rich = group[group[y_name] == 1]
                ratios.append(rich.shape[0] / n_group)
            else:
                # if group is empty, set ratio as 0
                ratios.append(0)
        
        # find the average ratio
        avg = np.mean(ratios)

        # for each group, replace either rich with poor or vice versa, so that the ratios are as similar as possible
        for i, group in enumerate(groups):
            rich = group[group[y_name] == 1]
            poor = group[group[y_name] == 0]
            n_rich = rich.shape[0]
            ratio = ratios[i]

            # either nobody is rich or nobody is poor: cannot duplicate people from an empty group
            if ratio == 0.0 or ratio == 1.0:
                print("group has only rich or only poor people, did not preprocess")
                continue

            # how many rows should be replaced (how far the group's ratio is from the average ratio)
            to_replace = int(np.abs(n_rich - (avg / ratio) * n_rich))

            # if more rich than average, replace rich with poor
            if ratio > avg:
                # randomly remove rich
                drop_indices = np.random.choice(rich.index, to_replace, replace=False)
                rich_preprocessed = rich.drop(drop_indices)

                # randomly duplicate poor
                duplicates = poor.sample(n=to_replace, replace=True)
                poor_preprocessed = pd.concat([poor, duplicates], axis=0)

            # if less rich than average, replace poor with rich
            else:
                # randomly remove poor
                drop_indices = np.random.choice(poor.index, to_replace, replace=False)
                poor_preprocessed = poor.drop(drop_indices)

                # randomly duplicate rich
                duplicates = rich.sample(n=to_replace, replace=True)
                rich_preprocessed = pd.concat([rich, duplicates], axis=0)
            
            # construct a new group with the correct amount of rich and poor
            group_preprocessed = pd.concat([rich_preprocessed, poor_preprocessed], axis=0)
            ratio_preprocessed = group_preprocessed[group_preprocessed[y_name] == 1].shape[0]/group_preprocessed.shape[0]

            print("ratio goal\t", avg)
            print("ratio after\t", ratio_preprocessed)
        
            # TODO replace the original group in the dataset with the preprocessed group

            # TODO return preprocessed dataset


print(preferential_resampling(adult_train, ['sex_ Male', 'sex_ Female'], 'hours-per-week', 'income'))
# TODO: try with other s than male and female

#%%

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_depth=20, random_state=0)
model.fit(train_X, train_Y)
pred = model.predict(test_X)
print(20, model.score(test_X, test_Y))

#%%
importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

feature_names = [f"feature {i}" for i in range(train_X.shape[1])]

forest_importances = pd.Series(importances, index=adult_train.drop('income', axis=1).columns)

forest_importances = forest_importances.sort_values(ascending=False)
print(forest_importances.head(6))

fig, ax = plt.subplots(1,1,figsize=(15,15))
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.savefig("feature_importances.jpg", dpi=800)


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
