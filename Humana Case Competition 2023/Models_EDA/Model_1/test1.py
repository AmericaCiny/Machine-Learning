import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('submission.csv')

# Calculate the mean score for each unique ID
mean_scores = df.groupby('id')['score'].transform('mean')

# Replace the 'score' column with the mean scores
df['score'] = mean_scores

# Add a 'rank' column to the DataFrame based on the 'mean_scores'
df.drop_duplicates(subset='id', keep='first', inplace=True)
df['rank'] = df['score'].rank(method='first', ascending=False).astype(int)
# Save the result to a new CSV file or print it
df.to_csv('result.csv', index=False)
