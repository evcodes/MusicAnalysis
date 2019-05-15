import re
import string

# PRE: Lyrics of a song in text format
# POST: Remove special characters in the text so the SA tokenizer does not get confused
def clean(lyrics):
	# split into list
	words = lyrics.split()

	# split by whitespace, remove punctuation
	table = str.maketrans('', '', string.punctuation)
	stripped_lyrics = ' '.join([w.translate(table) for w in words])

	return stripped_lyrics