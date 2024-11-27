import os
import pandas as pd

data_dir = 'THUCNews/data'
output_file = 'data.csv'

data = []

for filename in os.listdir(data_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(data_dir, filename)

        file_id = int(filename.split('.')[0]) 

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()

        data.append({'id': file_id, 'content': content})

df = pd.DataFrame(data)

df = df.sort_values(by='id').reset_index(drop=True)

df.to_csv(output_file, index=False, encoding='utf-8')

print(f"{output_file} have created!")s
