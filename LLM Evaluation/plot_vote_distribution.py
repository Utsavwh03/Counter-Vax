import csv
import matplotlib.pyplot as plt

# Read the CSV file
with open('/home/sohampoddar/HDD2/utsav/Evaluation/Survey_Data/Counter-Argument Evaluation Survey 0.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Get all columns that start with "Tweet" and don't contain "Explain"
tweet_columns = [col for col in data[0].keys() if col.startswith("Tweet") and "Explain" not in col]

# Initialize counters for each tweet
tweet_votes = {i: {'A': 0, 'B': 0} for i in range(1, 21)}

# Count votes for each tweet
for row in data:
    for col in tweet_columns:
        tweet_num = int(col.split()[1])
        if tweet_num <= 20:  # Only process first 20 tweets
            vote = row[col]
            if vote == 'Counter Argument A':
                tweet_votes[tweet_num]['A'] += 1
            elif vote == 'Counter Argument B':
                tweet_votes[tweet_num]['B'] += 1

# Create the plot
plt.figure(figsize=(15, 8))

# Prepare data for plotting
tweets = list(tweet_votes.keys())
votes_a = [tweet_votes[tweet]['A'] for tweet in tweets]
votes_b = [tweet_votes[tweet]['B'] for tweet in tweets]

# Create grouped bar plot
x = range(len(tweets))
width = 0.35

plt.bar([i - width/2 for i in x], votes_a, width, label='Counter Argument A', color='#2ecc71')
plt.bar([i + width/2 for i in x], votes_b, width, label='Counter Argument B', color='#e74c3c')

# Customize the plot
plt.title('Distribution of Votes for Counter Arguments A and B Across 20 Tweets', pad=20)
plt.xlabel('Tweet Number')
plt.ylabel('Number of Votes')
plt.legend(title='Counter Argument')
plt.xticks(x, tweets)

# Add value labels on top of bars
for i, (a, b) in enumerate(zip(votes_a, votes_b)):
    plt.text(i - width/2, a + 0.1, str(a), ha='center', va='bottom')
    plt.text(i + width/2, b + 0.1, str(b), ha='center', va='bottom')

# Adjust layout and save
plt.tight_layout()
plt.savefig('vote_distribution.png')
plt.close()