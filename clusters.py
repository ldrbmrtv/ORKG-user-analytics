from read import file, df
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
import json


def get_action_urls(x):
    urls = []
    for action in x:
        url = action.get('url')
        if url:
            n = url.count('/')
            if n >= 4:
                inds = [i for i, ch in enumerate(url) if ch == '/']
                last = inds[3]
                url = url[:last]
            if '?' in url:
                last = url.index('?')
                url = url[:last]
            url = url.replace('https://', '')
            url = url.replace('http://', '')
            url = url.replace('www.', '')
            time = action.get('timeSpent')
            if 'orkg' in url:
                urls.append({'url': url, 'time': time})
    urls = pd.DataFrame(urls)
    if not urls.empty:
        urls = urls.groupby('url').agg(sum)
        urls = urls['time'].to_dict()
    return urls


df['url'] = df['actionDetails'].apply(get_action_urls)
df = df[['url']]
df = pd.json_normalize(df['url'])
df.to_csv(f'{file}_post.csv')

df = df.fillna(0)
df = df.loc[:, df.var(ddof=0) > 0.1]
print(df.var(ddof=0))
scaler = MinMaxScaler()
X = scaler.fit_transform(df)
model = OPTICS(min_samples=10, metric='cosine')
model.fit(X)
print(len(set(model.labels_)))
df['label'] = model.labels_
df.to_csv(f'{file}_clusters.csv')

df_centers = df[df['label'] != -1]
df_centers = df_centers.groupby('label').agg('mean')
df_centers['count'] = df.groupby('label').size()
df_centers = df_centers.loc[:, df_centers.var(ddof=0) > 0.1]
print(df_centers.var(ddof=0))
df_centers = df_centers.sort_values(by='count', ascending=False)
df_centers.to_csv(f'{file}_centers.csv')

fig = plt.figure()
df_centers['count'].plot.bar()
plt.xlabel('Cluster')
plt.ylabel('Visits')
plt.tight_layout()
fig.savefig('counts.jpg')
plt.close()

max_time = 0
centers = df_centers.to_dict(orient='index')
for cluster, features in centers.items():
    new_features = {}
    for feature, time in features.items():
        if time >= 1:
            new_features[feature] = time
            if features['count'] >= 100 and feature != 'count' and time > max_time:
                print([feature, time])
                max_time = time        
    centers[cluster] = new_features

with open(f'{file}_centers.json', 'w') as file:
    json.dump(centers, file, indent=2)

for cluster, features in centers.items():
    count = features.pop('count')
    if count >= 100:
        fig = plt.figure(figsize=(10, 1.5*len(features)))
        plt.xlim(0, max_time)
        bar = plt.barh(list(features.keys()),
                       list(features.values()))
        plt.title(f'Cluster {cluster}, {count} users')
        plt.ylabel('Page')
        plt.xlabel('Time')
        plt.tight_layout()
        fig.savefig(f'bars/{cluster}.jpg')
        plt.close()
