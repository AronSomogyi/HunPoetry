import pandas as pd
import os
import re
import nltk
#import itertools

# Define the source directorysource_dir = "poem_texts"  # Change this to your source path
source_dir = 'poem_texts/'

# Init a df
data = []
for i in os.listdir(source_dir):
    poet_name = re.sub(r"_(.*)", "", i)
    if poet_name in [
        'Ady', 'AranyJ', 'Babits', 'Balassi', 'Csokonai', 'Jozsef', 'Karinthy', 'Kolcsey', 'Kosztolanyi',
        'Madach', 'Petofi', 'Radnoti', 'Vorosmarty']:

        for files in os.listdir(f"{source_dir}/{i}"):
            ### valami
            
            with open(f"{source_dir}/{i}/{files}", "r") as f:
                poem_text = f.read()

            data.append({'poet' : poet_name, 'poem' : poem_text})

# Remove numberings of verses
pattern = r'^(I+|II+|III+|IV+|V+|VI+|VII+|VIII+|IX+|X+|XI+|XII+| XIII+ | XIV+ | XV+ |[1-9][0-9]*[.)]?\s*)'

# DataFrame management
df = pd.DataFrame(data)
df['Cim'] = df.iloc[:, 1].str.split('\n').str[0]

# Remove Title from poems
for i in range(len(df)):
    df.loc[i,'poem'] = df.loc[i,'poem'].replace(df.loc[i, 'Cim'], '').strip()

df['poem'] = df['poem'].str.replace(pattern, '', regex=True).str.strip().str.lower()

df['poem'] = df['poem'].apply(lambda x: [word for word in x.split() if word not in (nltk.corpus.stopwords.words('hungarian'))])
   
df['poem'] = df['poem'].apply(lambda x: [re.sub(r'[^\w\s]', '', wrds) for wrds in x])


df.to_csv('df_preprocessed.csv')
