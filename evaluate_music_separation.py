
import argparse
import json
import torch
import os
import numpy as np
import scipy.io.wavfile as wavfile
import statistics as stat
from utils import si_sdr, beamformer_max_di, \
    beamformer_max_re, beamformer_max_sdr, zen_to_ele, azi_to_0_2pi_range
from network import DemucsDirection, center_trim
from pathlib import Path


def flatten(t):
    return [item for sublist in t for item in sublist]


def save_audio(save_folder, method, azi_angle, zen_angle, waveform):
    Path(save_folder).mkdir(parents = True, exist_ok = True)
    output_path = os.path.join(save_folder, method + '_waveform_azi_' + "{:.2f}".format(
        azi_angle * 180 / np.pi) + '_zen_' + "{:.2f}".format(zen_angle * 180 / np.pi) + '.wav')
    aux_waveform = waveform * np.iinfo(np.int16).max
    wavfile.write(output_path, 44100, aux_waveform.astype(np.int16))


def forward_pass(model, mixed_data, conditioning_direction, beamformer_audio, args):
    ambi_mixes = mixed_data.float().unsqueeze(0)  # Batch size is 1
    ambi_mixes = ambi_mixes.to(args.device)
    conditioning_direction = conditioning_direction.float().unsqueeze(0)
    conditioning_direction = conditioning_direction.to(args.device)
    beamformer_audio = beamformer_audio.float().unsqueeze(0)
    beamformer_audio = beamformer_audio.to(args.device)

    output_signal = model(ambi_mixes, conditioning_direction, beamformer_audio)
    output_signal = center_trim(output_signal, ambi_mixes)
    output_signal = torch.squeeze(output_signal, dim = 1)

    output_np = output_signal.detach().cpu().numpy()

    return output_np


def forward_beamformer(bf_type, input_signal, aux):
    if bf_type == 'max_di':
        beamformer = beamformer_max_di
    if bf_type == 'max_re':
        beamformer = beamformer_max_re
    if bf_type == 'max_sdr':
        beamformer = beamformer_max_sdr

    return beamformer(input_signal, aux)


def get_items(curr_dir, ambiorder):
    with open(Path(curr_dir) / 'metadata.json') as json_file:
        metadata = json.load(json_file)

    # Iterate over different sources
    source_positions = []
    source_audios = []
    for key in sorted(metadata.keys()):

        # get source audio
        gt_audio_files = sorted(
            list(Path(curr_dir).rglob(key + ".wav")))
        assert (len(gt_audio_files) > 0)
        _, gt_waveform = wavfile.read(gt_audio_files[0])
        gt_waveform = gt_waveform.astype(np.float)
        is_all_zero = np.all((gt_waveform == 0))
        if not is_all_zero:
            rms = np.sqrt(np.mean(gt_waveform ** 2))
            gt_waveform = gt_waveform * (0.1 / rms)  # desired rms is 0.1

        gt_waveform = gt_waveform.T.copy()  # MxT numpy array
        source_audios.append(gt_waveform)

        # get source position
        source_azi_angle = metadata[key]['panning_angles'][0]
        source_zen_angle = metadata[key]['panning_angles'][1]
        source_positions.append([source_azi_angle, source_zen_angle])

    # get mixture
    mix_path = os.path.join(curr_dir, "mix.wav")
    rate, mixture_waveform = wavfile.read(mix_path)
    mixture_waveform = mixture_waveform.astype(np.float)
    mix_is_all_zero = np.all((mixture_waveform[:, 0] == 0))
    if not mix_is_all_zero:
        mixture_waveform = mixture_waveform / np.amax(np.abs(mixture_waveform[:, 0])) / np.sqrt(2 * ambiorder + 1)

    return mixture_waveform, source_positions, source_audios, sorted(metadata.keys())


def main(args):
    print("result path will be")
    print(args.result_dir)
    print('\n')

    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')

    args.device = device
    n_channels = (args.ambiorder + 1) ** 2
    if args.ambimode == 'implicit':
        model = DemucsDirection(n_audio_channels = (args.ambiorder + 1) ** 2, ambimode = args.ambimode)
    elif args.ambimode == 'mixed':
        model = DemucsDirection(n_audio_channels = 5, ambimode = args.ambimode)

    model.load_state_dict(torch.load(args.model_checkpoint), strict = True)
    model.train = False
    model.to(device)

    all_dirs = sorted(list(Path(args.test_dir).glob('[0-9]*')))

    si_sdr_nn = {'vocals': [], 'drums': [], 'bass': []}
    si_sdr_beamformer_max_di = {'vocals': [], 'drums': [], 'bass': []}
    si_sdr_beamformer_max_re = {'vocals': [], 'drums': [], 'bass': []}
    si_sdr_beamformer_max_sdr = {'vocals': [], 'drums': [], 'bass': []}
    si_sdr_omnimix = {'vocals': [], 'drums': [],
                      'bass': []}  # For baseline, we consider the omni mix as the separated source

    si_sdr_stats = {'median': {'nn': {'vocals': None, 'drums': None, 'bass': None, 'all': None},
                               'beamformer_max_di': {'vocals': None, 'drums': None, 'bass': None, 'all': None},
                               'beamformer_max_re': {'vocals': None, 'drums': None, 'bass': None, 'all': None},
                               'beamformer_max_sdr': {'vocals': None, 'drums': None, 'bass': None, 'all': None},
                               'omnimix': {'vocals': None, 'drums': None, 'bass': None, 'all': None}}}

    checkpoint_folder = os.path.split(os.path.normpath(args.model_checkpoint))[0]
    if args.save_audios:
        print("Audios will be saved here:")
        print(os.path.join(checkpoint_folder, 'test_audio_samples'))
        print('\n')

    cmt = 0

    for idx in range(0, len(all_dirs)):
        print(idx)

        curr_dir = all_dirs[idx]

        # Loads the data
        mixed_data, source_positions, source_audios, sources_name = get_items(curr_dir, args.ambiorder)

        mixed_data = mixed_data[:, 0:n_channels]

        idx_folder = os.path.basename(os.path.normpath(curr_dir))
        omni_mix = mixed_data[:, 0]
        if args.save_audios:
            save_folder = os.path.join(checkpoint_folder, 'test_musdb_audio_samples', idx_folder)
            Path(save_folder).mkdir(parents = True, exist_ok = True)
            output_path = os.path.join(save_folder, 'omnimix.wav')
            aux_omni_mix = omni_mix * np.iinfo(np.int16).max
            wavfile.write(output_path, 44100, aux_omni_mix.astype(np.int16))

        for [azi_angle, zen_angle], gt_waveform, key in zip(source_positions, source_audios, sources_name):

            if args.save_audios:
                idx_folder = os.path.basename(os.path.normpath(curr_dir))
                save_folder = os.path.join(checkpoint_folder, 'test_musdb_audio_samples', idx_folder)
                save_audio(save_folder, 'gt', azi_angle, zen_angle, gt_waveform)

            azi_angle_beamformer = azi_to_0_2pi_range(azi_angle)
            ele_angle_beamformer = zen_to_ele(zen_angle)

            # beamformer_max_di output at this location
            beamformer_max_di_audio = forward_beamformer('max_di', mixed_data,
                                                         np.array((azi_angle_beamformer, ele_angle_beamformer)))[:, 0]
            if args.save_audios:
                save_audio(save_folder, 'max_di', azi_angle, zen_angle, beamformer_max_di_audio)

            # beamformer_max_re output at this location
            beamformer_max_re_audio = forward_beamformer('max_re', mixed_data,
                                                         np.array((azi_angle_beamformer, ele_angle_beamformer)))[:, 0]
            beamformer_audio = beamformer_max_re_audio.copy()
            beamformer_audio = np.expand_dims(beamformer_audio, axis=0)

            if args.save_audios:
                save_audio(save_folder, 'max_re', azi_angle, zen_angle, beamformer_max_re_audio)

            # beamformer_max_sdr output at this location
            beamformer_max_sdr_audio, singular_matrix = forward_beamformer('max_sdr', mixed_data, gt_waveform)
            if singular_matrix:
                cmt += 1
                print("Total singular matrices = " + str(cmt))
                print('\n')
            if args.save_audios and not singular_matrix:
                save_audio(save_folder, 'max_sdr', azi_angle, zen_angle, beamformer_max_sdr_audio)

            # neural network audio output at this location
            azi_normalized = (azi_angle + np.pi) / np.pi - 1
            zen_normalized = 2 * zen_angle / np.pi - 1
            conditioning_direction = np.asarray([azi_normalized, zen_normalized])
            conditioning_direction = torch.tensor(conditioning_direction).float()

            nn_mixed_data = mixed_data

            if args.ambimode == 'mixed':

                rms = np.sqrt(np.mean(beamformer_audio ** 2))
                if rms != 0:
                    beamformer_audio = beamformer_audio * (0.1 / rms)

                nn_mixed_data = mixed_data[:, 0:4]

            nn_mixed_data = torch.tensor(nn_mixed_data.T).float()
            nn_beamformer_audio = torch.tensor(beamformer_audio).float()

            nn_predicted_audio = forward_pass(model, nn_mixed_data, conditioning_direction, nn_beamformer_audio, args)
            nn_predicted_audio = nn_predicted_audio[0, 0, :]

            if args.save_audios:
                save_audio(save_folder, 'nn_' + args.ambimode, azi_angle, zen_angle, nn_predicted_audio)

            is_all_zero = np.all((gt_waveform == 0))
            if not is_all_zero:
                si_sdr_nn[key].append(si_sdr(nn_predicted_audio, gt_waveform))
                si_sdr_beamformer_max_di[key].append(si_sdr(beamformer_max_di_audio, gt_waveform))
                si_sdr_beamformer_max_re[key].append(si_sdr(beamformer_max_re_audio, gt_waveform))
                if not singular_matrix:
                    si_sdr_beamformer_max_sdr[key].append(si_sdr(beamformer_max_sdr_audio, gt_waveform))
                si_sdr_omnimix[key].append(si_sdr(omni_mix, gt_waveform))

    Path(args.result_dir).mkdir(parents = True, exist_ok = True)

    json_path = os.path.join(args.result_dir, 'si_sdr_nn.json')
    with open(json_path, 'w') as fp:
        json.dump(si_sdr_nn, fp)

    json_path = os.path.join(args.result_dir, 'si_sdr_beamformer_max_di.json')
    with open(json_path, 'w') as fp:
        json.dump(si_sdr_beamformer_max_di, fp)

    json_path = os.path.join(args.result_dir, 'si_sdr_beamformer_max_re.json')
    with open(json_path, 'w') as fp:
        json.dump(si_sdr_beamformer_max_re, fp)

    json_path = os.path.join(args.result_dir, 'si_sdr_beamformer_max_sdr.json')
    with open(json_path, 'w') as fp:
        json.dump(si_sdr_beamformer_max_sdr, fp)

    json_path = os.path.join(args.result_dir, 'si_sdr_omnimix.json')
    with open(json_path, 'w') as fp:
        json.dump(si_sdr_omnimix, fp)

    for key in ['vocals', 'drums', 'bass']:
        si_sdr_stats['median']['nn'][key] = stat.median(si_sdr_nn[key])
        si_sdr_stats['median']['beamformer_max_di'][key] = stat.median(si_sdr_beamformer_max_di[key])
        si_sdr_stats['median']['beamformer_max_re'][key] = stat.median(si_sdr_beamformer_max_re[key])
        si_sdr_stats['median']['beamformer_max_sdr'][key] = stat.median(si_sdr_beamformer_max_sdr[key])
        si_sdr_stats['median']['omnimix'][key] = stat.median(si_sdr_omnimix[key])

    si_sdr_stats['median']['nn']['all'] = stat.median(
        flatten([si_sdr_nn['vocals'], si_sdr_nn['drums'], si_sdr_nn['bass']]))
    si_sdr_stats['median']['beamformer_max_di']['all'] = stat.median(flatten(
        [si_sdr_beamformer_max_di['vocals'], si_sdr_beamformer_max_di['drums'], si_sdr_beamformer_max_di['bass']]))
    si_sdr_stats['median']['beamformer_max_re']['all'] = stat.median(flatten(
        [si_sdr_beamformer_max_re['vocals'], si_sdr_beamformer_max_re['drums'], si_sdr_beamformer_max_re['bass']]))
    si_sdr_stats['median']['beamformer_max_sdr']['all'] = stat.median(flatten(
        [si_sdr_beamformer_max_sdr['vocals'], si_sdr_beamformer_max_sdr['drums'], si_sdr_beamformer_max_sdr['bass']]))
    si_sdr_stats['median']['omnimix']['all'] = stat.median(
        flatten([si_sdr_omnimix['vocals'], si_sdr_omnimix['drums'], si_sdr_omnimix['bass']]))

    json_path = os.path.join(args.result_dir, 'si_sdr_stats.json')
    with open(json_path, 'w') as fp:
        json.dump(si_sdr_stats, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir', type = str,
                        help = "Path to the testing directory")
    parser.add_argument('model_checkpoint', type = str,
                        help = "Path to the model file")
    parser.add_argument('--use_cuda', dest = 'use_cuda', action = 'store_true',
                        help = "Whether to use cuda")
    parser.add_argument('--ambiorder', type = int, default = 4,
                        help = "Ambisonics order")
    parser.add_argument('--ambimode', type = str, default = 'implicit',
                        help = "Ambisonics mode. 'implicit': raw Ambisonics mixture as input. "
                               "'mixed': raw first order Ambisonics mixture and bf concatenated.")
    parser.add_argument('--save_audios', dest = 'save_audios', action = 'store_true',
                        help = "Whether to save predicted audios")
    parser.add_argument('--result_dir', dest = 'result_dir', type = str,
                        help = "Path for the si_sdr results")

    print(parser.parse_args())
    main(parser.parse_args())
