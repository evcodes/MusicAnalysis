
import lyricwikia
from clean_lyrics import clean

def get_lyrics(artist_name, song_name):
    
    try:
        lyrics = lyricwikia.get_lyrics(artist_name, song_name)
        return clean(lyrics)
    except:
        print("Sorry, " + song_name +" by " +  artist_name + " was not found in lyric database")
        return ""

print(get_lyrics("Eminem", "lose yourself"))