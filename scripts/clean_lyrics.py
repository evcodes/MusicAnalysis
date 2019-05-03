import re
import string

def clean_lyrics(lyrics="Hello my name is ^^^^what my na*&me is"):
	# split into list
	lyrics = "Hello my name is ^^^^what my na*&me is"
	words = lyrics.split()

	# split by whitespace, remove punctuation
	table = str.maketrans('', '', string.punctuation)
	stripped_lyrics = ' '.join([w.translate(table) for w in words])

	return stripped_lyrics