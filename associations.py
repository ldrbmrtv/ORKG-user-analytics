from read import file, df
from itertools import pairwise
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = df[['actionDetails']]
df['actionDetails'] = df['actionDetails'].apply(lambda x: [y.get('url') for y in x])
df['actionDetails'] = df['actionDetails'].apply(lambda x: list(pairwise(x)))
df = df.explode('actionDetails')
df = df.dropna()
df = pd.DataFrame(df['actionDetails'].to_list(), columns=['action1', 'action2'])
df = df[df['action1'] != df['action2']]
df = df.groupby(['action1', 'action2'], as_index = False).size()
df = df.sort_values(by = 'size', ascending = False)
#df = df[df['size'] > 30]
print(df.head())
print(df.info())
df.to_csv('transitions.csv')

pages = df['action1'].unique()
df_heat = pd.DataFrame(None, index=pages, columns=pages)
for index, row in df.iterrows():
    df_heat[row['action1']][row['action2']] = row['size']
df_heat = df_heat.fillna(0)
df_heat['sum'] = df_heat.sum(axis=1)
df_heat = df_heat.sort_values(by = 'sum', ascending=False)
df_heat = df_heat.drop(columns = ['sum'])
#df_heat = df_heat[df_heat.index]
df_heat.loc['sum'] = df.sum()
df_heat = df_heat.sort_values(by = 'sum', ascending=False, axis=1)
df_heat = df_heat.drop(index = ['sum'])
d = 20
df_heat = df_heat.head(d)
df_heat = df_heat[df_heat.columns.to_list()[:d]]
df_heat = df_heat.T
df_heat.to_csv('transitions_map.csv')
fig = plt.figure(figsize=(12, 10))
sns.heatmap(df_heat)
plt.tight_layout()
fig.savefig('map.jpg')
plt.close()


df = df.groupby('action1').agg({'action2': lambda x: list(x), 'size': lambda x: list(x)})
df['sum'] = df['size'].apply(sum)
df = df.sort_values(by = 'sum', ascending = False)
df['size'] = df.apply(lambda x: [round(int(y)/x['sum'], 3) for y in x['size']], axis=1)
df['actions'] = [{} for x in df.index]
for index, row in df.iterrows():
    for action, size in zip(row['action2'], row['size']):
        row['actions'][action] = size
df = df.drop(columns = ['action2', 'size'])
df.to_json('transitions_grouped.json', orient = 'index', indent = 2)
