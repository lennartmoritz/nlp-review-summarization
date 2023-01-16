from bertopic import BERTopic
import pandas as pd

# Read CSV data to dataframe
df = pd.read_csv("C:/Users/Lennart/Documents/Uni/NLP/Crazy-NLP-Dropbox/data.csv", encoding='unicode_escape')
titles = df["movie_title"].unique()
print(f"There are {len(titles)} different entries for movie_title.")
print("Among the titles are:")
print(titles[:10])

# Only select reviews for a single movie
print(df.info())
df = df.loc[df['movie_title'] == titles[1]]
total = len(df)
print(f"There are {total} reviews of {titles[1]}.")
print(df.head(3))

# convert pd.nan entries from float to string to work with model
docs = list(df.loc[:, "review_content"].astype('str').values)

# Limit processing to up to 1000 reviews
docs = docs[:1000]
# print(docs[:3])

# Execute the topic modeling
print(f"Executing topic modelling for {len(docs)} reviews...")
assert len(docs)
topic_model = BERTopic(language="english")
topics, probs = topic_model.fit_transform(docs)

# Print topic results. The -1 topic refers to all outlier documents and is typically ignored
topic_info = topic_model.get_topic_info()
example_topic = topic_model.get_topic(1)
print(topic_info)
print(example_topic)
