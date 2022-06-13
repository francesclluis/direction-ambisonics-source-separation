
import argparse
import multiprocessing
import os
import json
import numpy as np
import torch
import torch.optim as optim
import scipy.io.wavfile as wavfile
from pathlib import Path
from dataset import Dataset
from network import DemucsDirection, center_trim, load_pretrain


def train_epoch(model, device, optimizer, train_loader, epoch, log_interval=20):
    # Set the model to training.
    model.train()

    # Training loop
    losses = []
    interval_losses = []

    for batch_idx, (ambi_mixes, target_signals,
                    target_direction, beamformer_audio) in enumerate(train_loader):
        ambi_mixes = ambi_mixes.to(device)
        target_signals = target_signals.to(device)
        target_direction = target_direction.to(device)
        beamformer_audio = beamformer_audio.to(device)

        # Reset grad
        optimizer.zero_grad()

        output_signal = model(ambi_mixes, target_direction, beamformer_audio)
        output_signal = center_trim(output_signal, ambi_mixes)

        output_signal = torch.squeeze(output_signal, dim = 1)

        loss = model.loss(output_signal, target_signals)

        interval_losses.append(loss.item())

        # Backpropagation
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        # Update the weights
        optimizer.step()

        # Print the loss
        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}".format(
                epoch, batch_idx * len(ambi_mixes), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                np.mean(interval_losses)))

            losses.extend(interval_losses)
            interval_losses = []

    return np.mean(losses)


def test_epoch(model, device, test_loader, args, epoch, log_interval=20):
    model.eval()
    test_loss = 0
    output_folder = os.path.join(args.checkpoints_dir, args.name, 'samples')

    with torch.no_grad():

        for batch_idx, (ambi_mixes, target_signals,
                        target_direction, beamformer_audio) in enumerate(test_loader):
            ambi_mixes = ambi_mixes.to(device)
            ambi_mixes_original = ambi_mixes
            target_signals = target_signals.to(device)
            target_direction = target_direction.to(device)
            beamformer_audio = beamformer_audio.to(device)

            # Run through the model
            output_signal = model(ambi_mixes, target_direction, beamformer_audio)
            output_signal = center_trim(output_signal, ambi_mixes)

            output_signal = torch.squeeze(output_signal, dim = 1)

            if batch_idx == 0 and epoch % 10 == 0:
                for b in range(output_signal.shape[0]):
                    output_signal_np = output_signal.detach().cpu().numpy()
                    target_signals_np = target_signals.detach().cpu().numpy()
                    ambi_mixes_original_np = ambi_mixes_original.detach().cpu().numpy()

                    output_signal_np = output_signal_np * np.iinfo(np.int16).max
                    target_signals_np = target_signals_np * np.iinfo(np.int16).max
                    ambi_mixes_original_np = ambi_mixes_original_np * np.iinfo(np.int16).max

                    wavfile.write(os.path.join(output_folder,
                                               'epoch_' + str(epoch) + '_batch_pos_' + str(b) + '_output_signal.wav'),
                                  args.sr, output_signal_np[b, ...].T.astype(np.int16))
                    wavfile.write(os.path.join(output_folder, 'epoch_' + str(epoch) + '_batch_pos_' + str(
                        b) + '_label_source_signals.wav'), args.sr, target_signals_np[b, ...].T.astype(np.int16))
                    wavfile.write(os.path.join(output_folder,
                                               'epoch_' + str(epoch) + '_batch_pos_' + str(b) + '_input_mixture.wav'),
                                  args.sr, ambi_mixes_original_np[b, ...].T.astype(np.int16))

            loss = model.loss(output_signal, target_signals)
            test_loss += loss.item()

            if batch_idx % log_interval == 0:
                print("Loss: {}".format(loss))

        test_loss /= len(test_loader)
        print("\nTest set: Average Loss: {:.4f}\n".format(test_loss))

        return test_loss


def train(args):
    # Load dataset
    if args.dataset == 'musdb':
        args.sr = 44100
    if args.dataset == 'fuss':
        args.sr = 16000
    data_train = Dataset(args.train_dir, sr = args.sr, ambiorder = args.ambiorder,
                         angular_window_deg = 2.5, ambimode = args.ambimode, dataset = args.dataset)
    data_test = Dataset(args.test_dir, sr = args.sr, ambiorder = args.ambiorder,
                        angular_window_deg = 2.5, ambimode = args.ambimode, dataset = args.dataset)

    # Set up the device and workers.
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print("Using device {}".format('cuda' if use_cuda else 'cpu'))

    # Set multiprocessing params
    num_workers = min(multiprocessing.cpu_count(), args.n_workers)
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': True
    } if use_cuda else {}

    # Set up data loaders
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size = args.batch_size,
                                               shuffle = True, **kwargs)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size = args.batch_size,
                                              **kwargs)

    # Set up model
    print('SETTING UP MODEL')
    if args.ambimode == 'implicit':
        model = DemucsDirection(n_audio_channels = (args.ambiorder + 1) ** 2, ambimode = args.ambimode)
    elif args.ambimode == 'mixed':
        model = DemucsDirection(n_audio_channels = 5, ambimode = args.ambimode)

    model.to(device)

    print('MODEL SET UP')
    # Set up checkpoints
    if not os.path.exists(os.path.join(args.checkpoints_dir, args.name)):
        os.makedirs(os.path.join(args.checkpoints_dir, args.name))
    if not os.path.exists(os.path.join(args.checkpoints_dir, args.name, 'samples')):
        os.makedirs(os.path.join(args.checkpoints_dir, args.name, 'samples'))

    # Save commandline_args
    commandline_args_path = os.path.join(args.checkpoints_dir, args.name, 'commandline_args.txt')
    with open(commandline_args_path, 'w') as f:
        json.dump(args.__dict__, f, indent = 2)

    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr = args.lr,
                           weight_decay = args.decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 10, verbose = True)

    # Load pretrain
    if args.pretrain_path:
        print('LOADING PRETRAINED')
        state_dict = torch.load(args.pretrain_path)
        load_pretrain(model, state_dict)
        print('PRETRAINED LOADED')

    # Load the model if `args.start_epoch` is greater than 0. This will load the model from
    # epoch = `args.start_epoch - 1`
    if args.start_epoch is not None:
        assert args.start_epoch > 0, "start_epoch must be greater than 0."
        start_epoch = args.start_epoch
        checkpoint_path = Path(
            args.checkpoints_dir) / "{}.pt".format(start_epoch - 1)
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)
    else:
        start_epoch = 0

    # Loss values
    best_error = float("inf")
    train_losses = []
    test_losses = []

    loss_dict = {'train': [], 'test': []}

    print('GOING TO TRAINING LOOP')
    # Training loop
    try:
        for epoch in range(start_epoch, args.epochs + 1):

            train_loss = train_epoch(model, device, optimizer, train_loader,
                                     epoch, args.print_interval)
            torch.save(
                model.state_dict(),
                os.path.join(args.checkpoints_dir, args.name, "last.pt"))

            print("Done with training, going to testing")
            test_loss = test_epoch(model, device, test_loader, args, epoch,
                                   args.print_interval)
            if test_loss < best_error:
                best_error = test_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(args.checkpoints_dir, args.name, "best.pt"))

            scheduler.step(test_loss)

            train_losses.append((epoch, train_loss))
            test_losses.append((epoch, test_loss))

            # save json
            loss_dict['train'].append(train_loss)
            loss_dict['test'].append(test_loss)
            json_path = os.path.join(args.checkpoints_dir, args.name, 'loss.json')
            with open(json_path, 'w') as fp:
                json.dump(loss_dict, fp)

        return train_losses, test_losses

    except KeyboardInterrupt:
        # print("Interrupted")
        import traceback  # pylint: disable=import-outside-toplevel
        traceback.print_exc()
    except Exception as _:  # pylint: disable=broad-except
        import traceback  # pylint: disable=import-outside-toplevel
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data Params
    parser.add_argument('train_dir', type = str,
                        help = "Path to the training dataset")
    parser.add_argument('test_dir', type = str,
                        help = "Path to the testing dataset")
    parser.add_argument('--name', type = str, default = "multimic_experiment",
                        help = "Name of the experiment")
    parser.add_argument('--checkpoints_dir', type = str, default = './checkpoints',
                        help = "Path to the checkpoints")
    parser.add_argument('--batch_size', type = int, default = 8,
                        help = "Batch size")
    parser.add_argument('--ambiorder', type = int, default = 4,
                        help = "Ambisonics order")
    parser.add_argument('--ambimode', type = str, default = 'implicit',
                        help = "Ambisonics mode. 'implicit': raw Ambisonics mixture as input. "
                               "'mixed': raw first order Ambisonics mixture and bf concatenated.")
    parser.add_argument('--dataset', type = str, default = "musdb",
                        help = "Dataset to train")

    # Training Params
    parser.add_argument('--epochs', type = int, default = 350,
                        help = "Number of epochs")
    parser.add_argument('--lr', type = float, default = 1e-4, help = "learning rate")
    parser.add_argument('--sr', type = int, default = 44100, help = "Sampling rate")
    parser.add_argument('--decay', type = float, default = 0, help = "Weight decay")
    parser.add_argument('--n_workers', type = int, default = 16,
                        help = "Number of parallel workers")
    parser.add_argument('--print_interval', type = int, default = 20,
                        help = "Logging interval")
    parser.add_argument('--start_epoch', type = int, default = None,
                        help = "Start epoch")
    parser.add_argument('--pretrain_path', type = str,
                        help = "Path to pretrained weights")
    parser.add_argument('--use_cuda', dest = 'use_cuda', action = 'store_true',
                        help = "Whether to use cuda")

    train(parser.parse_args())
