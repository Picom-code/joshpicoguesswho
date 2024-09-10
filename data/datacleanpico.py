import pandas as pd

# Load the dataset
data = pd.read_csv('PICO.csv')  # Adjust the path to your CSV file

# Drop the specified columns
data = data.drop(['Frowning', 'MiddlePart', 'Kid', 'Hat','Glasses','LongHair'], axis=1)
#middlepart, kid, hat, glasses, long hair, frowning
# Save the modified dataset if necessary
data.to_csv('PICO.csv', index=False)

# Now you can continue with your analysis or training...
