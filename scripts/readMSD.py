import h5py
import numpy as np

import hdf5_getters



song_dict = {

    'artist_mbid'
    'artist_mbtags'
    'artist_mbtags_count'
    'artist_name'
    'artist_playmeid'
    'artist_terms'
    'artist_terms_freq'
    'artist_terms_weight'
    'audio_md5'
    'bars_confidence'
    'bars_start'
    'beats_confidence'
    'beats_start'
    'danceability'
    'duration'
    'end_of_fade_in'
    'energy'
    'key'
    'key_confidence'
    'loudness'
    'mode'
    'mode_confidence'
    'release'
    'release_7digitalid'
    'sections_confidence'
    'sections_start'
    'segments_confidence'
    'segments_loudness_max'
    'segments_loudness_max_time'
    'segments_loudness_start'
    'segments_pitches'
    'segments_start'
    'segments_timbre'
    'similar_artists'
    'song_hotttnesss'
    'song_id'
    'start_of_fade_out'
    'tatums_confidence'
    'tatums_start'
    'tempo'
    'time_signature'
    'time_signature_confidence'
    'title'
    'track_7digitalid'
    'track_id'
    'year'
    'artist_gender'
    'genre'
    'billboard_pos'
    'billboard_date'
    'lyrical_sa'
    
}


def artist(filename):
    h5 = hdf5_getters.open_h5_file_read(filename)
    duration = hdf5_getters.get_artist_name(h5)
    h5.close()
    return duration.decode('utf-8')
    
def song_name(filename):
    h5 = hdf5_getters.open_h5_file_read(filename)
    song = hdf5_getters.get_title(h5)
    h5.close()
    return song.decode('utf-8')



artist('../Databases/MSDsub/A/A/A/TRAAAAW128F429D538.h5')
print(song_name('../Databases/MSDsub/A/A/A/TRAAAAW128F429D538.h5'))



# h = h5py.File('../Databases/MSDsub/A/A/A/TRAAAAW128F429D538.h5', 'r+')

# print(list(h.keys()))

# keys = []
# for i in h.keys():
#     keys.append(i)

# for k in keys:
#     for j in h[k]:
#         print(h[k][j])
#         print(h[k][j].value)
# # df1= np.array(X1)

# # for j in (X1):
#     # print(X1[j])
#     # print(X1[j].value)
