import pandas as pd

INPUT_FILE_PATH = './data/All-seasons.csv'
OUPUT_FILE_PATH = './data/train_data.csv'


df = pd.read_csv(INPUT_FILE_PATH)

clean_lines = pd.Series(
    [filter_lines
        .replace('\n', '')
        .replace('(', '')
        .replace(')', '')
        .replace('  ', ' ')
        .strip()
        for filter_lines in df.Line
     ]
)

train_data = pd.DataFrame(df.Character)
del df

train_data['line'] = clean_lines

train_data.columns = ['name', 'line']


train_data.to_csv(OUPUT_FILE_PATH, index=False)
