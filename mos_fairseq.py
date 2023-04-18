# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

import os
import sys
import argparse
import fairseq
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import random

try:
    # Data2VecMultiModelを参照できるように、@register_modelされたclassをimportする必要あり
    sys.path.append(os.environ["D2V2_SPECTROGRAM_PYTHONPATH"])
    import examples.data2vec.models.data2vec2  # noqa: E402, F401
    import examples.data2vec.tasks.spectrogram_pretraining  # noqa: E402, F401
except (KeyError, ModuleNotFoundError) as e:
    print(e)
    print("Module data2vec2 is not used")

try:
    # Data2VecMultiModelを参照できるように、@register_modelされたclassをimportする必要あり
    sys.path.append(os.environ["MAE_AST_PYTHONPATH"])
    import mae_ast.models.mae_ast  # noqa: E402, F401
    import mae_ast.tasks.mae_ast_pretraining  # noqa: E402, F401
except (KeyError, ModuleNotFoundError) as e:
    print(e)
    print("Module mae_ast is not used")

try:
    # spectrogramの処理も必要
    sys.path.append(os.environ["TRANSFORM_TO_SPECTROGRAM_PYTHONPATH"])
    from transform_to_spectrogram import transform_to_spectrogram  # noqa: E402, F401
except (KeyError, ModuleNotFoundError) as e:
    print(e)
    print("transform_to_spectrogram")


SSL_OUT_DIM = int(os.environ["SSL_OUT_DIM"])
INPUT_TYPE = os.getenv("INPUT_TYPE", "wav")
WIN_LENGTH = int(os.getenv("WIN_LENGTH", 400))
HOP_LENGTH = int(os.getenv("HOP_LENGTH", 160))
N_FFT = int(os.getenv("N_FFT", 512))
OUTPUT_CHANNEL_DIM = int(os.getenv("OUTPUT_CHANNEL_DIM", 128))


class MosPredictor(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim):
        super(MosPredictor, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.output_layer = nn.Linear(self.ssl_features, 1)

    def forward(self, wav):
        wav = wav.squeeze(1)  ## [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res["x"]
        x = torch.mean(x, 1)
        x = self.output_layer(x)
        return x.squeeze(1)


class MyDataset(Dataset):
    def __init__(self, wavdir, mos_list):
        self.mos_lookup = {}
        f = open(mos_list, "r")
        for line in f:
            parts = line.strip().split(",")
            wavname = parts[0]
            mos = float(parts[1])
            self.mos_lookup[wavname] = mos

        self.wavdir = wavdir
        self.wavnames = sorted(self.mos_lookup.keys())

    def __getitem__(self, idx):
        wavname = self.wavnames[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        wav = torchaudio.load(wavpath)[0]
        score = self.mos_lookup[wavname]
        return wav, score, wavname

    def __len__(self):
        return len(self.wavnames)

    def collate_fn(self, batch):  ## zero padding
        wavs, scores, wavnames = zip(*batch)
        wavs = list(wavs)
        max_len = max(wavs, key=lambda x: x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), "constant", 0)
            if INPUT_TYPE != "wav":
                padded_wav = transform_to_spectrogram(
                    padded_wav,
                    INPUT_TYPE,
                    win_length=WIN_LENGTH,
                    hop_length=HOP_LENGTH,
                    n_fft=N_FFT,
                    output_channel_dim=OUTPUT_CHANNEL_DIM,
                )
            output_wavs.append(padded_wav)

        output_wavs = torch.stack(output_wavs, dim=0)
        scores = torch.stack([torch.tensor(x) for x in list(scores)], dim=0)
        return output_wavs, scores, wavnames


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir", type=str, required=True, help="Path of your DATA/ directory"
    )
    parser.add_argument(
        "--fairseq_base_model",
        type=str,
        required=True,
        help="Path to pretrained fairseq base model",
    )
    parser.add_argument(
        "--finetune_from_checkpoint",
        type=str,
        required=False,
        help="Path to your checkpoint to finetune from",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=False,
        default="checkpoints",
        help="Output directory for your trained checkpoints",
    )
    parser.add_argument("--seed", type=int, required=False, default=0, help="seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    cp_path = args.fairseq_base_model
    datadir = args.datadir
    ckptdir = args.outdir
    my_checkpoint = args.finetune_from_checkpoint

    if not os.path.exists(ckptdir):
        os.system("mkdir -p " + ckptdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: " + str(device))

    wavdir = os.path.join(datadir, "wav")
    trainlist = os.path.join(datadir, "sets/train_mos_list.txt")
    validlist = os.path.join(datadir, "sets/val_mos_list.txt")

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()

    trainset = MyDataset(wavdir, trainlist)
    trainloader = DataLoader(
        trainset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=trainset.collate_fn,
        worker_init_fn=seed_worker,
        generator=g,
    )

    validset = MyDataset(wavdir, validlist)
    validloader = DataLoader(
        validset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=validset.collate_fn,
        worker_init_fn=seed_worker,
        generator=g,
    )

    net = MosPredictor(ssl_model, SSL_OUT_DIM)
    net = net.to(device)

    if my_checkpoint != None:  ## do (further) finetuning
        net.load_state_dict(torch.load(my_checkpoint))

    criterion = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    PREV_VAL_LOSS = 9999999999
    orig_patience = 20
    patience = orig_patience
    for epoch in range(1, 1001):
        STEPS = 0
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels, filenames = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            STEPS += 1
            running_loss += loss.item()
        print("EPOCH: " + str(epoch))
        print("AVG EPOCH TRAIN LOSS: " + str(running_loss / STEPS))
        epoch_val_loss = 0.0
        net.eval()
        ## clear memory to avoid OOM
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        ## validation
        VALSTEPS = 0
        for i, data in enumerate(validloader, 0):
            VALSTEPS += 1
            inputs, labels, filenames = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / VALSTEPS
        print("EPOCH VAL LOSS: " + str(avg_val_loss))
        if avg_val_loss < PREV_VAL_LOSS:
            print("Loss has decreased")
            PREV_VAL_LOSS = avg_val_loss
            PATH = os.path.join(ckptdir, "ckpt_" + str(epoch))
            torch.save(net.state_dict(), PATH)
            PATH = os.path.join(ckptdir, "ckpt_best")
            torch.save(net.state_dict(), PATH)
            patience = orig_patience
        else:
            patience -= 1
            if patience == 0:
                print(
                    "loss has not decreased for "
                    + str(orig_patience)
                    + " epochs; early stopping at epoch "
                    + str(epoch)
                )
                break

    print("Finished Training")


if __name__ == "__main__":
    main()
