import lyricwikia
from clean_lyrics import clean

# PRE: Given an artist name and song title
# POST: Return the song lyrics if it exists in database, else return Error.
def get_lyrics(artist_name, song_name):
    
    try:
        lyrics = lyricwikia.get_lyrics(artist_name, song_name)
        return clean(lyrics)
    except:
        print("Sorry, " + song_name +" by " +  artist_name + " was not found in lyric database")
        return ""

print(get_lyrics("Eminem", "lose yourself"))