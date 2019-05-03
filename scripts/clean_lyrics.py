import re
import string

def clean(lyrics):
	# split into list
	words = lyrics.split()

	# split by whitespace, remove punctuation
	table = str.maketrans('', '', string.punctuation)
	stripped_lyrics = ' '.join([w.translate(table) for w in words])

	return stripped_lyrics