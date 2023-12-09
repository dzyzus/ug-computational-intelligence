import nltk
import matplotlib.pyplot as plt
import bs4
import urllib.request
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

link = "https://www.bbc.co.uk/news/world-europe-67667035"

webpage=str(urllib.request.urlopen(link).read())
soup = bs4.BeautifulSoup(webpage)

article_text = soup.get_text()

print(f"\nLoaded text: \n{article_text}\n")

# uncomment if u need more packages from ntlk
# nltk.download()

# tokenize words
sent_tokens = sent_tokenize(article_text)
word_tokens = word_tokenize(article_text)

print(f"\nSent token {sent_tokens}\nWord token {word_tokens}")

# init stop_words
stop_words = set(stopwords.words('english'))

# elimnate stop words
filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

# with no lower case conversion
filtered_sentence = []
 
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(f"work tokens after delete stop words: \n{word_tokens}")
print(f"\nFiltered sentence\n{filtered_sentence}")

# Examples of lemmatization:

# how it works
#-> rocks : rock
#-> corpora : corpus
#-> better : good

# !!!!!!!!!! ADD ADDITIONAL BAD_WORDS <- TODO

stop_words.add(".")
stop_words.add(", ")
stop_words.add("'")
stop_words.add("'s")
stop_words.add("``")
stop_words.add(",")
stop_words.add(":")
stop_words.add("''")

# init word lemmatizer
lemmatizer = WordNetLemmatizer()

lemmatized_words = []

# Lemmatize `word` using WordNet's built-in morphy function
for w in word_tokens:
    if w not in stop_words:
        lemmatized_words.append(w)

print(f"Count of words after lemmatize operation {lemmatized_words.count}")

# Count top 10 most popular words
fdist = FreqDist(lemmatized_words)

top_words = fdist.most_common(10)
words, counts = zip(*top_words)

plt.bar(words, counts)
plt.xlabel('Word')
plt.ylabel('Freq')
plt.title('Top 10 the most frequently words in text')
plt.show()

# Generate a word cloud
wordcloud = WordCloud(stopwords=stop_words, background_color="white", width=800, height=400).generate(' '.join(lemmatized_words))

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()