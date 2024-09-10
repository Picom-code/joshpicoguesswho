import pandas as pd
data = pd.read_csv('PICO.csv') 
data = data.drop(['Frowning', 'MiddlePart', 'Kid', 'Hat','Glasses','LongHair'], axis=1)
data.to_csv('PICO.csv', index=False)


