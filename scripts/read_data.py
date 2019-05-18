import pandas as pd
from hdf5_getters import *
import os
import glob
from collections import defaultdict
from pprint import pprint as pp
from lyric_api import get_lyrics
from sentiment import get_sent, prep_sent
import numpy as np

# Things we need from other databases:
# artist_gender
# genre
# billboard_pos
# billboard_date
# lyrical_sa
attribute_list = []


def create_artist_gender_dict():
    # extract artist names + corresponding gender info
    singers_gender_data = pd.read_csv('../Databases/singers_gender.csv',
                                      encoding="ISO-8859-1")
    artist_names = singers_gender_data['artist']
    artist_gender = singers_gender_data['gender']

    # create dictionary from names and dictionary
    gender_names = pd.concat([artist_names, artist_gender], axis=1)
    gender_names = gender_names.set_index('artist')
    gender_names_dict = gender_names.to_dict()['gender']

    return gender_names_dict


def to_csv(songs):
    with open("../Databases/songs.csv", 'w') as f:
        for i, att_title in enumerate(attribute_list):
            if i < len(attribute_list) - 1:
                f.write(att_title[0] + ",")
            else:
                f.write(att_title[0] + "\n")
        for song in songs.keys():
            atts = songs[song].items()
            for i, att in enumerate(atts):
                if i < len(atts) - 1:
                    f.write(att + ",")
                else:
                    f.write(att + "\n")


def get_artist_gender(artist_name, gender_names_dict):
    # if artist_name.decode("utf-8") in gender_names_dict:
    # 	return gender_names_dict[artist_name.decode("utf-8")]
    # else:
    # 	return 'Gender Unknown'
    return gender_names_dict[artist_name.decode("utf-8")]


def get_lyric_sentiment(artist_name, song_name):
    lyrics = get_lyrics(str(artist_name), str(song_name))
    if lyrics != None:
        lyric_list = [i for i in lyrics.split() if i.isalnum()]
        lyric_len = len(lyric_list)
        sent = []
        for i in range((lyric_len // 30) + 1):
            lyr = lyric_list[30 * i:min((30 * i) + 30, lyric_len)]
            lyr = " ".join(lyr)
            sent.append(np.array(get_sent(lyr)))

        result = [np.mean(k, axis=0) for k in zip(*sent)]
        result = np.ndarray.tolist(result[0])
        print(result)
        return result
    else:
        return [0, 0, 0, 0, 0]


# map artist names and genders to all attribute_lists from Million Songs into one dictionary
def create_song_records(basedir, ext='.h5'):
    gender_names_dict = create_artist_gender_dict()

    attribute_list.append(('artist_name', get_artist_name))
    attribute_list.append(('gender', get_artist_gender))
    attribute_list.append(('artist_mbid', get_artist_mbid))
    attribute_list.append(('artist_mbtags', get_artist_mbtags))
    attribute_list.append(('artist_mbtags_count', get_artist_mbtags_count))
    # attribute_list.append(('artist_name', get_artist_name))
    attribute_list.append(('artist_playmeid', get_artist_playmeid))
    attribute_list.append(('artist_terms', get_artist_terms))
    attribute_list.append(('artist_terms_freq', get_artist_terms_freq))
    attribute_list.append(('audio_md5', get_audio_md5))
    attribute_list.append(('bars_confidence', get_bars_confidence))
    attribute_list.append(('bars_start', get_bars_start))
    attribute_list.append(('beats_confidence', get_beats_confidence))
    attribute_list.append(('beats_start', get_beats_start))
    attribute_list.append(('danceability', get_danceability))
    attribute_list.append(('duration', get_duration))
    attribute_list.append(('end_of_fade_in', get_end_of_fade_in))
    attribute_list.append(('energy', get_energy))
    attribute_list.append(('key', get_key))
    attribute_list.append(('key_confidence', get_key_confidence))
    attribute_list.append(('loudness', get_loudness))
    attribute_list.append(('mode', get_mode))
    attribute_list.append(('mode_confidence', get_mode_confidence))
    attribute_list.append(('release_7digitalid', get_release_7digitalid))
    attribute_list.append(('sections_confidence', get_sections_confidence))
    attribute_list.append(('sections_start', get_sections_start))
    attribute_list.append(('segments_confidence', get_segments_confidence))
    attribute_list.append(('segments_loudness_max', get_segments_loudness_max))
    attribute_list.append(('segments_loudness_max_time',
                           get_segments_loudness_max_time))
    attribute_list.append(('segments_loudness_start',
                           get_segments_loudness_start))
    attribute_list.append(('segments_pitches', get_segments_pitches))
    attribute_list.append(('segments_start', get_segments_start))
    attribute_list.append(('segments_timbre', get_segments_timbre))
    attribute_list.append(('similar_artists', get_similar_artists))
    attribute_list.append(('song_hotttnesss', get_song_hotttnesss))
    attribute_list.append(('song_id', get_song_id))
    attribute_list.append(('start_of_fade_out', get_start_of_fade_out))
    attribute_list.append(('tatums_confidence', get_tatums_confidence))
    attribute_list.append(('tatums_start', get_tatums_start))
    attribute_list.append(('tempo', get_tempo))
    attribute_list.append(('time_signature', get_time_signature))
    attribute_list.append(('time_signature_confidence',
                           get_time_signature_confidence))
    attribute_list.append(('title', get_title))
    attribute_list.append(('track_7digitalid', get_track_7digitalid))
    attribute_list.append(('track_id', get_track_id))
    attribute_list.append(('year', get_year))
    attribute_list.append(('sentiment', get_lyric_sentiment))

    song_records = defaultdict(dict)
    dircounter = 0
    filecounter = 0
    attcounter = 0
    for root, dirs, files in os.walk(basedir):
        dircounter += 1
        print("dircounter", dircounter)
        files = glob.glob(os.path.join(root, '*' + ext))

        for f in files:
            filecounter += 1
            print("filecounter", filecounter)
            h5 = open_h5_file_read(f)
            song_id = get_song_id(h5)

            # Make an entry only if the artist name from MSD exists in the gender_names_dict
            if get_artist_name(h5).decode("utf-8") in gender_names_dict:
                # For testing
                # print("\n\n\nSong ID: ", song_id.decode("utf-8"))

                for i in attribute_list:
                    attcounter += 1
                    print(attcounter, i[0])
                    if (i[0] == 'gender'):
                        song_records[song_id.decode("utf-8")][i[0]] = \
                            i[1](get_artist_name(h5), gender_names_dict)
                    elif i[0] == 'sentiment':
                        song_sent = i[1](get_artist_name(h5).decode("utf-8"),
                                         get_title(h5).decode("utf-8"))
                        song_records[song_id.decode("utf-8")][i[0]] = song_sent
                    else:
                        song_records[song_id.decode("utf-8")][i[0]] = i[1](h5)

                # For testing
                # print("\n", i[0], ": ", song_records[song_id.decode("utf-8")][i[0]])
                h5.close()

            else:
                h5.close()
                continue

            # For testing
            print("\n\n\n")

    print(type(song_records), "keys:", len(song_records.keys()))
    # pp(song_records)
    return song_records


def main():
    prep_sent()
    basedir = '../Databases/MSDsub/'
    song_records = create_song_records(basedir)
    to_csv(song_records)


if __name__ == "__main__":
    main()
