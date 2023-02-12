import pandas as pd
df = pd.read_csv('./data/All-seasons.csv')
cleanlines = pd.Series(
    [cell
        .replace('\n', '')
        .replace('(', '')
        .replace(')', '')
        .replace('  ', ' ')
        .strip()
        for cell in df.Line
     ]
)

train = pd.DataFrame(df.Character)
train['line'] = cleanlines
train.columns = ['name', 'line']

train.to_csv('./data/train.csv', index=False)
