

filename = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv'

data = pd.read_csv(filename, header=None, na_values='?')

#%%
data = data.dropna()

print(data.shape)

#%%

occupationcounts = dict(data[6].value_counts())


#%%

import matplotlib.pyplot as plt

fig = plt.figure(figsize= (25, 10))

plt.bar(occupationcounts.keys, occupationcounts.values)
plt.xticks(fontsize=20, rotation = 45)