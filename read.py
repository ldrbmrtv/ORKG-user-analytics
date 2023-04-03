import pandas as pd
        

file = 'visits_log'
df = pd.read_json(f'{file}.json', orient='records')
print(df.info())
