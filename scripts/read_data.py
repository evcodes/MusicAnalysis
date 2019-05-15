import pandas as pd
import numpy as np
import h5py
from hdf5_getters import *
import os
import glob
from collections import defaultdict

# extract artist names + corresponding gender info
singers_gender_data = pd.read_csv('../Databases/singers_gender.csv', encoding="ISO-8859-1")
artist_names   = singers_gender_data['artist']
artist_gender = singers_gender_data['gender']

# Things we need from other databases:
	# artist_gender
	# genre
	# billboard_pos
	# billboard_date
	# lyrical_sa


# map artist names and genders to all attribute_lists from Million Songs into one table
def create_song_record(basedir, ext='.h5'):

	attribute_list = []
	attribute_list.append(('artist_mbid', get_artist_mbid))
	attribute_list.append(('artist_mbtags', get_artist_mbtags))
	attribute_list.append(('artist_mbtags_count', get_artist_mbtags_count))
	attribute_list.append(('artist_name', get_artist_name))
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
	attribute_list.append(('segments_loudness_max_time', get_segments_loudness_max_time))
	attribute_list.append(('segments_loudness_start', get_segments_loudness_start))
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
	attribute_list.append(('time_signature_confidence', get_time_signature_confidence))
	attribute_list.append(('title', get_title))
	attribute_list.append(('track_7digitalid', get_track_7digitalid))
	attribute_list.append(('track_id', get_track_id))
	attribute_list.append(('year', get_year))

	song_records = defaultdict(dict)

	for root, dirs, files in os.walk(basedir):
		
		files = glob.glob(os.path.join(root,'*'+ext))

		for f in files:
			h5        = open_h5_file_read(f)
			song_id   = get_song_id(h5)

			print(h5)
			for i in attribute_list:
				song_records[song_id.decode("utf-8")][i[0]] = i[1](h5)
			
			h5.close()

	return song_records


def main():
	basedir = '../Databases/MSDsub/'
	# get_titles(basedir, ext='.h5')
	create_song_record(basedir)

if __name__ == "__main__" :
	main()

















	