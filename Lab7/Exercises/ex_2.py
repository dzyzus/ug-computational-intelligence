from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from text2emotion import get_emotion

# pip install emoji==1.7.0.

negative_comment = "The mattresses were badly in need of cleaning or even replacement. It was disgusting. I left a night early. Additionally when I turned on the shower, a lot of hair that wasn’t mine came out of the drain, and I had to wash my feet again the moment I stepped on the bath mat because it was filthy."
positive_comment = "Very friendly and accommodating staff. There was enough storage room and the sink in the room was quite convenient. The location is in a nice neighborhood and super close to a metro station. It was possible to store our luggage before our check-in and after our check-out."

sentences = [negative_comment, positive_comment]

negative_com_tokenize = tokenize.sent_tokenize(negative_comment)

for sentence in sentences:
    sid = SentimentIntensityAnalyzer()
    emotions = get_emotion(sentence)
    print(f"\n{sentence}\n")
    ss = sid.polarity_scores(sentence)
    print("\nEmotions:\n")
    for emotion, score in emotions.items():
        print(f'{emotion}: {score}')

    print("\nSummary:\n")
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
        print()