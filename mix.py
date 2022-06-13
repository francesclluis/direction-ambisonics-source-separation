
import numpy as np
import scipy as sci
import numpy.random as rnd
import scipy.io.wavfile
import musdb
import argparse
import copy
from pathlib import Path
import os
import json
import librosa
from utils import eval_sh
from pyroom.pyroom import Roomsimulator
from pyroom.utility.Coordinates import Coordinates


def generateOneRandomSourcePosition(roomSize):
    position = roomSize.cart * (np.random.rand(3) * 2 - 1)

    # no sources on the floor or under the ceiling
    position[2] = roomSize.cart[2] * (np.random.rand(1) - 0.5)
    c = Coordinates(position)

    return c


def prepareMUSDB():
    root = '/musdb18hq'  # path to musdb18hq dataset

    # separates training set into training and validation
    fraction_validate_tracks = 0.9

    last_validate_track = None

    if subset == 'train':
        mus = musdb.DB(root = root, is_wav = True, subsets = "train")
        path = os.path.join(base_path, 'train_dir')
        num_tracks = len(mus)
        last_validate_track = int(num_tracks * fraction_validate_tracks)
    elif subset == 'validate':
        mus = musdb.DB(root = root, is_wav = True, subsets = "train")
        path = os.path.join(base_path, 'validate_dir')
        num_tracks = len(mus)
        last_validate_track = int(num_tracks * fraction_validate_tracks)
    elif subset == 'test':
        mus = musdb.DB(root = root, is_wav = True, subsets = "test")
        path = os.path.join(base_path, 'test_dir')
        num_tracks = len(mus)

    return num_tracks, last_validate_track, path, mus


def prepareFuss():
    root = '/sound-separation/datasets/fuss/fuss_data/fuss_dev/ssdata'  # path to FUSS ssdata

    if subset == 'train':
        read_path = os.path.join(root, 'train')
        write_path = os.path.join(base_path, 'train_dir')
    elif subset == 'validate':
        read_path = os.path.join(root, 'validation')
        write_path = os.path.join(base_path, 'validate_dir')
    elif subset == 'test':
        read_path = os.path.join(root, 'eval')
        write_path = os.path.join(base_path, 'test_dir')

    return read_path, write_path


# Set some global parameters
# Ambisoncs order
max_order = 4
num_sh_channels = (max_order + 1) ** 2

# Length of the samples
length_s = 6
ir_length_s = 1

# Variable input parameters
parser = argparse.ArgumentParser()
parser.add_argument("subset", help = "subset is train, validate or test")
parser.add_argument("num_mixes", help = "number of mixes created on one instance", type = int)
parser.add_argument("num_mixes_with_silent_sources",
                    help = "number of mixes with silent sources created on that instance", type = int)
parser.add_argument("minimal_angular_dist", help = "minimum angular distance between sources in degree", type = float)
parser.add_argument("base_path", help = "path for the resulting dataset")

parser.add_argument("--maximal_angular_dist",
                    help = "maximal angular distance between sources in degree (for generating closed sources dataset)",
                    type = float, default = 180.0)
parser.add_argument("--batch_index", help = "when running on multiple instances, this is the index of the instance",
                    type = int, default = 0)
parser.add_argument('--render_room', dest = 'render_room', action = 'store_true', default = False)
parser.add_argument("--dataset", help = "for now, musdb or fuss", type = str, default = 'musdb')
parser.add_argument("--level_threshold_db", help = "level threshold db for a mix not to count as silent", type = float,
                    default = '-60.0')

parser.add_argument("--room_size_range", help = "range of variation from the default room size in m", type = list,
                    default = [0, 0, 0], nargs = '+')
parser.add_argument("--rt_range", help = "range of variation from the default reverberation time", type = float,
                    default = 0)

# Parse input parameters
args = parser.parse_args()

subset = args.subset
num_mixes = args.num_mixes
num_mixes_with_silent_sources = args.num_mixes_with_silent_sources
minimal_angular_dist_deg = args.minimal_angular_dist
maximal_angular_dist_deg = args.maximal_angular_dist
base_path = args.base_path
batch_index = args.batch_index

render_room = args.render_room
room_size_range = np.array(args.room_size_range[0]).astype(np.float64)
rt_range = args.rt_range

level_threshold_db = args.level_threshold_db

dataset = args.dataset

# Default sampling rate and number of samples
if dataset == 'musdb':
    sampling_rate = 44100
    num_samples = length_s * sampling_rate

elif dataset == 'fuss':
    sampling_rate = 16000
    num_samples = length_s * sampling_rate

minimal_angular_dist_rad = float(minimal_angular_dist_deg) / 180 * np.pi
maximal_angular_dist_rad = float(maximal_angular_dist_deg) / 180 * np.pi

print(
    f'Starting dataset generation {dataset}, subset = {subset} \n number of mixes on this node = {num_mixes} '
    f'\n mixes with silent sources = {num_mixes_with_silent_sources} \n sample length = {num_samples} '
    f'\n result path = {base_path} \n room rendering {render_room}')

if render_room:
    roomSim = Roomsimulator()

    # Default Room Size
    default_room_size = Coordinates([3, 4, 3])

    # Default Reverberation time for [  125.   250.   500.  1000.  2000.  4000.  8000. 16000.] Hz
    default_rt = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])

    roomSim.fs = sampling_rate
    roomSim.maxShOrder = max_order
    # set some parameters
    roomSim.maxIsOrder = 6

    # prepare general room simulation
    roomSim.prepareImageSource()
    roomSim.prepareWallFilter()
    roomSim.plotWallFilters()

    roomSim.irLength_s = ir_length_s

    roomSim.alignDirectSoundToStart = True

iMix = 0
iMixWithSilentSources = 0

if dataset == 'musdb':

    num_tracks, last_test_track, path, mus = prepareMUSDB()

    ir_length_samp = ir_length_s * sampling_rate

    azi, zen, ele = np.zeros(3), np.zeros(3), np.zeros(3)

    while iMix < num_mixes:

        if (subset == 'train'):
            iTrack_vocal = rnd.randint(0, last_test_track)
            iTrack_drums = rnd.randint(0, last_test_track)
            iTrack_bass = rnd.randint(0, last_test_track)
        elif (subset == 'validate'):
            iTrack_vocal = rnd.randint(last_test_track, num_tracks - 1)
            iTrack_drums = rnd.randint(last_test_track, num_tracks - 1)
            iTrack_bass = rnd.randint(last_test_track, num_tracks - 1)
        elif (subset == 'test'):
            iTrack = rnd.randint(0, num_tracks)
            iTrack_vocal = iTrack
            iTrack_drums = iTrack
            iTrack_bass = iTrack

        # get the tracks
        x_vocals = mus[iTrack_vocal].sources['vocals'].audio
        x_drums = mus[iTrack_drums].sources['drums'].audio
        x_bass = mus[iTrack_bass].sources['bass'].audio

        # select random starting point
        offset_vocals = np.random.randint(0, x_vocals.shape[0] - num_samples)
        offset_drums = np.random.randint(0, x_drums.shape[0] - num_samples)
        offset_bass = np.random.randint(0, x_bass.shape[0] - num_samples)

        # for testset, get from the same starting point
        if (subset == 'test'):
            offset_drums = offset_vocals
            offset_bass = offset_vocals

        # make mono version
        x_vocals_mono = np.mean(x_vocals[offset_vocals:offset_vocals + num_samples, :], axis = 1)
        x_drums_mono = np.mean(x_drums[offset_drums:offset_drums + num_samples, :], axis = 1)
        x_bass_mono = np.mean(x_bass[offset_bass:offset_bass + num_samples, :], axis = 1)

        ## compute levels
        vocal_level_db = 20 * np.log10(np.sqrt(np.sum(x_vocals_mono ** 2) / num_samples))
        drums_level_db = 20 * np.log10(np.sqrt(np.sum(x_drums_mono ** 2) / num_samples))
        bass_level_db = 20 * np.log10(np.sqrt(np.sum(x_bass_mono ** 2) / num_samples))

        # only proceed with this snippet, if the levels are above a certain threshold
        if (((vocal_level_db > level_threshold_db) & (drums_level_db > level_threshold_db) & (
                bass_level_db > level_threshold_db)) or subset == 'test'):

            # if the number of mixes with silent sources hasn't been reached, silence a random source
            if (iMixWithSilentSources < num_mixes_with_silent_sources):
                idxSilent = np.random.randint(0, 3)
                if idxSilent == 0:
                    x_vocals_mono = x_vocals_mono * 0
                elif idxSilent == 1:
                    x_drums_mono = x_drums_mono * 0
                elif idxSilent == 2:
                    x_bass_mono = x_bass_mono * 0

                iMixWithSilentSources = iMixWithSilentSources + 1

            if render_room:
                # modify room size and reverberation time on each iteration
                room_size = Coordinates(default_room_size.cart + room_size_range * (np.random.rand(3) - 0.5) * 2)
                roomSim.roomSize = room_size
                source_position_range = Coordinates(room_size.cart / 2.0)
                roomSim.rt = default_rt + rt_range * (np.random.rand(8) - 0.5) * 2
            else:
                source_position_range = Coordinates(
                    [1, 1, 1])  # in case there is no room simulations, generate points in a cube

            # generate positions one by one, make sure that no two sources are closer than minimal_angular_dist_rad
            p1 = generateOneRandomSourcePosition(source_position_range)
            p2 = copy.deepcopy(p1)
            p3 = copy.deepcopy(p1)

            # Try placing another source, at least minimal_angular_dist_rad away from the first
            while (p1.greatCircleDistanceTo(p2) < minimal_angular_dist_rad or p1.greatCircleDistanceTo(
                    p2) > maximal_angular_dist_rad):
                p2 = generateOneRandomSourcePosition(source_position_range)

            # Try placing another source, at least minimal_angular_dist_rad away from the first two
            while (p1.greatCircleDistanceTo(p3) < minimal_angular_dist_rad) or p1.greatCircleDistanceTo(
                    p3) > maximal_angular_dist_rad or \
                    (p2.greatCircleDistanceTo(p3) < minimal_angular_dist_rad or p2.greatCircleDistanceTo(
                        p3) > maximal_angular_dist_rad):
                p3 = generateOneRandomSourcePosition(source_position_range)

            if render_room:

                ## Simulate for the first source
                roomSim.sourcePosition = p1
                srir1 = roomSim.simulate()

                x_vocals_ambi = np.zeros((num_samples + srir1.shape[0] - 1, num_sh_channels))
                for iShChannel in range(num_sh_channels):
                    x_vocals_ambi[:, iShChannel] = sci.signal.convolve(x_vocals_mono, srir1[:, iShChannel])

                ## Second Source
                roomSim.sourcePosition = p2
                srir2 = roomSim.simulate()

                x_drums_ambi = np.zeros((num_samples + srir2.shape[0] - 1, num_sh_channels))
                for iShChannel in range(num_sh_channels):
                    x_drums_ambi[:, iShChannel] = sci.signal.convolve(x_drums_mono, srir2[:, iShChannel])

                ## Third Source
                roomSim.sourcePosition = p3
                srir3 = roomSim.simulate()

                x_bass_ambi = np.zeros((num_samples + srir3.shape[0] - 1, num_sh_channels))
                for iShChannel in range(num_sh_channels):
                    x_bass_ambi[:, iShChannel] = sci.signal.convolve(x_bass_mono, srir3[:, iShChannel])

                x_vocals_mono = np.hstack((x_vocals_mono, np.zeros(ir_length_samp - 1)))
                x_drums_mono = np.hstack((x_drums_mono, np.zeros(ir_length_samp - 1)))
                x_bass_mono = np.hstack((x_bass_mono, np.zeros(ir_length_samp - 1)))

            else:
                # Mixes without room
                x_vocals_ambi = np.outer(x_vocals_mono, eval_sh(max_order, p1.aziEle))
                x_drums_ambi = np.outer(x_drums_mono, eval_sh(max_order, p2.aziEle))
                x_bass_ambi = np.outer(x_bass_mono, eval_sh(max_order, p3.aziEle))

            # the best mix ever
            x_mix = (x_vocals_ambi + x_drums_ambi + x_bass_ambi) / 3

            # normalize to the maximal entry
            x_mix = x_mix / np.max(np.abs(x_mix))

            # scale for 16-bit wav file
            x_mix_scaled = x_mix * np.iinfo(np.int16).max
            x_vocals_scaled = x_vocals_mono * np.iinfo(np.int16).max
            x_drums_scaled = x_drums_mono * np.iinfo(np.int16).max
            x_bass_scaled = x_bass_mono * np.iinfo(np.int16).max

            # create new folder with sample iTrack
            output_prefix_dir = os.path.join(path, '{:05d}'.format(iMix + batch_index * num_mixes))
            Path(output_prefix_dir).mkdir(parents = True, exist_ok = True)

            # write the audios to folder
            # Ambisonics mix
            output_path_mix = os.path.join(output_prefix_dir, 'mix.wav')
            scipy.io.wavfile.write(output_path_mix, sampling_rate,
                                   x_mix_scaled.astype(np.int16))
            # dry mono files
            output_path_vocals = os.path.join(output_prefix_dir, 'vocals.wav')
            scipy.io.wavfile.write(output_path_vocals, sampling_rate,
                                   x_vocals_scaled.astype(np.int16))
            # dry mono files
            output_path_drums = os.path.join(output_prefix_dir, 'drums.wav')
            scipy.io.wavfile.write(output_path_drums, sampling_rate,
                                   x_drums_scaled.astype(np.int16))
            # dry mono files
            output_path_bass = os.path.join(output_prefix_dir, 'bass.wav')
            scipy.io.wavfile.write(output_path_bass, sampling_rate,
                                   x_bass_scaled.astype(np.int16))

            if render_room:
                output_path_srir = os.path.join(output_prefix_dir, 'srir1.wav')
                srir_int = srir1 * np.iinfo(np.int16).max
                scipy.io.wavfile.write(output_path_srir, roomSim.fs, srir_int.astype(np.int16))
                output_path_srir = os.path.join(output_prefix_dir, 'srir2.wav')
                srir_int = srir2 * np.iinfo(np.int16).max
                scipy.io.wavfile.write(output_path_srir, roomSim.fs, srir_int.astype(np.int16))
                output_path_srir = os.path.join(output_prefix_dir, 'srir3.wav')
                srir_int = srir3 * np.iinfo(np.int16).max
                scipy.io.wavfile.write(output_path_srir, roomSim.fs, srir_int.astype(np.int16))

            azi = np.array([p1.azi, p2.azi, p3.azi])
            zen = np.array([p1.zen, p2.zen, p3.zen])

            azi_normalized = (azi + np.pi) % (2 * np.pi) - np.pi
            dir_sph = np.vstack((azi_normalized, zen))

            metadata = {}
            metadata['vocals'] = {
                'panning_angles': dir_sph[:, 0].tolist(),
                'position_cartesian': p1.cart.tolist(),
                'original_track_index': iTrack_vocal,
            }
            metadata['drums'] = {
                'panning_angles': dir_sph[:, 1].tolist(),
                'position_cartesian': p2.cart.tolist(),
                'original_track_index': iTrack_drums,

            }
            metadata['bass'] = {
                'panning_angles': dir_sph[:, 2].tolist(),
                'position_cartesian': p3.cart.tolist(),
                'original_track_index': iTrack_bass,
            }

            metadata_file = str(Path(output_prefix_dir) / "metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent = 4)

            # if a room is rendered, also include information about that
            if render_room:
                room_metadata = {}
                room_metadata['room'] = {
                    'room_size': roomSim.roomSize.cart.tolist(),
                    'rt': roomSim.rt.tolist(),
                }

                room_metadata_file = str(Path(output_prefix_dir) / "room_metadata.json")
                with open(room_metadata_file, "w") as f:
                    json.dump(room_metadata, f, indent = 4)

            iMix = iMix + 1
            print('iMix: ' + str(iMix))
            print('\n')

elif dataset == 'fuss':

    num_samples = length_s * sampling_rate
    ir_length_samp = ir_length_s * sampling_rate

    read_path, write_path = prepareFuss()

    for root, subdirectories, files in os.walk(read_path):

        aux_dir = {}
        for subdirectory in subdirectories:
            print(os.path.join(root, subdirectory))
            curr_path = os.path.join(root, subdirectory)
            filenames = [filename for filename in os.listdir(curr_path) if filename.endswith('.wav')]

            if render_room:
                # modify room size and reverberation time on each iteration
                room_size = Coordinates(default_room_size.cart + room_size_range * (np.random.rand(3) - 0.5) * 2)
                roomSim.roomSize = room_size
                # get the maximal possible source position range
                distance_from_walls_m = 0.2
                source_position_range = Coordinates(room_size.cart / 2.0 - distance_from_walls_m)

                roomSim.rt = default_rt + rt_range * (np.random.rand(8) - 0.5) * 2
            else:
                source_position_range = Coordinates(
                    [1, 1, 1])  # in case there is no room simulations, generate points in a cube

            p0 = generateOneRandomSourcePosition(
                source_position_range)  # start by setting the same position to each source

            sources = [{'filename': filename, 'position': p0, 'azi': p0.azi, 'zen': p0.zen, 'mono_audio': None,
                        'ambi_audio': None, 'level': -np.inf} for filename in filenames]

            for s_num, s in enumerate(sources):

                # get the tracks. wavfile read sources
                source_audio, _ = librosa.load(os.path.join(curr_path, s['filename']), sr = sampling_rate,
                                               mono = True)  # Esta a 16 kHz el de fuss...

                # find a segment, in which the source is above a defined threshold
                cmt = 0
                while s['level'] < level_threshold_db:
                    cmt += 1
                    offset_source = np.random.randint(0, source_audio.shape[0] - num_samples)
                    segment_source = source_audio[offset_source:offset_source + num_samples]
                    s['mono_audio'] = segment_source

                    if cmt == 4000:
                        print('Filename: ' + s['filename'])
                        print('After 4000 iterations, no segment was found below the level threshold of: ' + str(
                            level_threshold_db) + ' dB.')
                        break

                    if not (np.sum(s['mono_audio']) == 0):
                        s['level'] = 20 * np.log10(np.sqrt(np.sum(s['mono_audio'] ** 2) / num_samples))

                print(f"this mix has {s_num + 1} sources")

                if s_num == 0:
                    pass
                else:
                    if s_num == 1:
                        p1 = copy.deepcopy(p0)
                        # if there are two sources, find a position for the second source, that fulfills the requirements,
                        # keep generating if these requirements are not met:
                        while (p0.greatCircleDistanceTo(p1) < minimal_angular_dist_rad or p0.greatCircleDistanceTo(
                                p1) > maximal_angular_dist_rad):
                            p1 = generateOneRandomSourcePosition(source_position_range)
                        s['position'] = p1
                        s['azi'] = p1.azi
                        s['zen'] = p1.zen

                    if s_num == 2:
                        p2 = copy.deepcopy(p1)
                        # try placing a potential third source
                        while (p1.greatCircleDistanceTo(p2) < minimal_angular_dist_rad) or p1.greatCircleDistanceTo(
                                p2) > maximal_angular_dist_rad or \
                                (p0.greatCircleDistanceTo(p2) < minimal_angular_dist_rad or p0.greatCircleDistanceTo(
                                    p2) > maximal_angular_dist_rad):
                            p2 = generateOneRandomSourcePosition(source_position_range)
                        s['position'] = p2
                        s['azi'] = p2.azi
                        s['zen'] = p2.zen

                    if s_num == 3:
                        # Try placing another source, at least windowsize_rad away from the first three
                        p3 = copy.deepcopy(p2)
                        # try placing a potential fourth source
                        while (p2.greatCircleDistanceTo(p3) < minimal_angular_dist_rad) or p2.greatCircleDistanceTo(
                                p3) > maximal_angular_dist_rad or \
                                (p1.greatCircleDistanceTo(p3) < minimal_angular_dist_rad) or p1.greatCircleDistanceTo(
                            p3) > maximal_angular_dist_rad or \
                                (p0.greatCircleDistanceTo(p3) < minimal_angular_dist_rad or p0.greatCircleDistanceTo(
                                    p3) > maximal_angular_dist_rad):
                            p3 = generateOneRandomSourcePosition(source_position_range)

                        s['position'] = p3
                        s['azi'] = p3.azi
                        s['zen'] = p3.zen

                # normalize azi between [-pi, pi]
                s['azi'] = (s['azi'] + np.pi) % (2 * np.pi) - np.pi

                if render_room:

                    # run room simulation
                    roomSim.sourcePosition = s['position']
                    srir = roomSim.simulate()

                    # init ambi mix
                    s['ambi_audio'] = np.zeros((num_samples + srir.shape[0] - 1, num_sh_channels))
                    for iShChannel in range(num_sh_channels):
                        s['ambi_audio'][:, iShChannel] = sci.signal.convolve(s['mono_audio'], srir[:, iShChannel])

                else:
                    s['ambi_audio'] = np.outer(s['mono_audio'], eval_sh(max_order, [s['azi'], np.pi / 2 - s[
                        'zen']]))  # eval_sh works with ele

                output_prefix_dir = os.path.join(write_path, '{:05d}'.format(iMix + batch_index * num_mixes))
                print('output_prefix_dir :' + str(output_prefix_dir))
                Path(output_prefix_dir).mkdir(parents = True, exist_ok = True)

            all_ambi_signals = []
            for s in sources:
                all_ambi_signals.append(s['ambi_audio'])

            all_ambi_signals = np.stack(all_ambi_signals, axis = 0)
            x_mix = np.sum(all_ambi_signals, axis = 0) / len(filenames)

            # normalize to the maximal entry
            x_mix = x_mix / np.max(np.abs(x_mix))

            # save mixture
            x_mix_scaled = x_mix * np.iinfo(np.int16).max
            output_path_mix = os.path.join(output_prefix_dir, 'mix.wav')
            scipy.io.wavfile.write(output_path_mix, sampling_rate, x_mix_scaled.astype(np.int16))
            # save each source
            metadata = {}
            metadata['num_sources'] = len(filenames)
            for s_num, s in enumerate(sources):
                if render_room:
                    s['mono_audio'] = np.hstack(
                        (s['mono_audio'], np.zeros(ir_length_samp - 1)))  # pad to same length as reverberant mixture
                s_scaled = s['mono_audio'] * np.iinfo(np.int16).max
                output_path_source = os.path.join(output_prefix_dir, 'source_' + str(s_num) + '.wav')
                scipy.io.wavfile.write(output_path_source, sampling_rate, s_scaled.astype(np.int16))
                metadata[s_num] = {'panning_angles': [s['azi'], s['zen']]}

                if render_room:
                    output_path_srir = os.path.join(output_prefix_dir, 'srir' + s['filename'] + '.wav')
                    srir_int = srir * np.iinfo(np.int16).max
                    scipy.io.wavfile.write(output_path_srir, roomSim.fs, srir_int.astype(np.int16))

            metadata_file = str(Path(output_prefix_dir) / "metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent = 4)

            # if a room is rendered, also include information about that
            if render_room:
                room_metadata = {}
                room_metadata['room'] = {
                    'room_size': roomSim.roomSize.cart.tolist(),
                    'rt': roomSim.rt.tolist(),
                }
                room_metadata_file = str(Path(output_prefix_dir) / "room_metadata.json")
                with open(room_metadata_file, "w") as f:
                    json.dump(room_metadata, f, indent = 4)

            iMix = iMix + 1
            print('iMix: ' + str(iMix))
            print('\n')
