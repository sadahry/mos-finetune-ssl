# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

import os
import sys
import argparse
import torch
import torch.nn as nn
import fairseq
from torch.utils.data import DataLoader
from mos_fairseq import MosPredictor, MyDataset
import numpy as np
import scipy.stats

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


def systemID(uttID):
    return uttID.split("-")[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fairseq_base_model",
        type=str,
        required=True,
        help="Path to pretrained fairseq base model.",
    )
    parser.add_argument(
        "--datadir", type=str, required=True, help="Path of your DATA/ directory"
    )
    parser.add_argument(
        "--finetuned_checkpoint",
        type=str,
        required=True,
        help="Path to finetuned MOS prediction checkpoint.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=False,
        default="answer.txt",
        help="Output filename for your answer.txt file for submission to the CodaLab leaderboard.",
    )
    args = parser.parse_args()

    fairseq_base_model = args.fairseq_base_model
    if fairseq_base_model.lower() == "none":
        fairseq_base_model = None
    my_checkpoint = args.finetuned_checkpoint
    datadir = args.datadir
    outfile = args.outfile

    system_csv_path = os.path.join(datadir, "mydata_system.csv")

    ssl_model = None
    if fairseq_base_model is not None:
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [fairseq_base_model]
        )
        ssl_model = model[0]
        ssl_model.remove_pretraining_modules()

    print("Loading checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MosPredictor(ssl_model).to(device)
    model.eval()

    model.load_state_dict(torch.load(my_checkpoint))

    wavdir = os.path.join(datadir, "wav")
    validlist = os.path.join(datadir, "sets/val_mos_list.txt")

    print("Loading data")
    validset = MyDataset(wavdir, validlist)
    validloader = DataLoader(
        validset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        collate_fn=validset.collate_fn,
    )

    total_loss = 0.0
    num_steps = 0.0
    predictions = {}  # filename : prediction
    criterion = nn.L1Loss()
    print("Starting prediction")

    for i, data in enumerate(validloader, 0):
        inputs, labels, filenames = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        output = outputs.cpu().detach().numpy()[0]
        predictions[filenames[0]] = output  ## batch size = 1

    true_MOS = {}
    validf = open(validlist, "r")
    for line in validf:
        parts = line.strip().split(",")
        uttID = parts[0]
        MOS = float(parts[1])
        true_MOS[uttID] = MOS

    ## compute correls.
    sorted_uttIDs = sorted(predictions.keys())
    ts = []
    ps = []
    for uttID in sorted_uttIDs:
        t = true_MOS[uttID]
        p = predictions[uttID]
        ts.append(t)
        ps.append(p)

    truths = np.array(ts)
    preds = np.array(ps)

    ### UTTERANCE
    MSE = np.mean((truths - preds) ** 2)
    print("[UTTERANCE] Test error= %f" % MSE)
    LCC = np.corrcoef(truths, preds)
    print("[UTTERANCE] Linear correlation coefficient= %f" % LCC[0][1])
    SRCC = scipy.stats.spearmanr(truths.T, preds.T)
    print("[UTTERANCE] Spearman rank correlation coefficient= %f" % SRCC[0])
    KTAU = scipy.stats.kendalltau(truths, preds)
    print("[UTTERANCE] Kendall Tau rank correlation coefficient= %f" % KTAU[0])

    ### SYSTEM
    true_sys_MOS_avg = {}
    csv_file = open(system_csv_path, "r")
    csv_file.readline()  ## skip header
    for line in csv_file:
        parts = line.strip().split(",")
        sysID = parts[0]
        MOS = float(parts[1])
        true_sys_MOS_avg[sysID] = MOS

    pred_sys_MOSes = {}
    for uttID in sorted_uttIDs:
        sysID = systemID(uttID)
        noop = pred_sys_MOSes.setdefault(sysID, [])
        pred_sys_MOSes[sysID].append(predictions[uttID])

    pred_sys_MOS_avg = {}
    for k, v in pred_sys_MOSes.items():
        avg_MOS = sum(v) / (len(v) * 1.0)
        pred_sys_MOS_avg[k] = avg_MOS

    ## make lists sorted by system
    pred_sysIDs = sorted(pred_sys_MOS_avg.keys())
    sys_p = []
    sys_t = []
    for sysID in pred_sysIDs:
        sys_p.append(pred_sys_MOS_avg[sysID])
        sys_t.append(true_sys_MOS_avg[sysID])

    sys_true = np.array(sys_t)
    sys_predicted = np.array(sys_p)

    MSE = np.mean((sys_true - sys_predicted) ** 2)
    print("[SYSTEM] Test error= %f" % MSE)
    LCC = np.corrcoef(sys_true, sys_predicted)
    print("[SYSTEM] Linear correlation coefficient= %f" % LCC[0][1])
    SRCC = scipy.stats.spearmanr(sys_true.T, sys_predicted.T)
    print("[SYSTEM] Spearman rank correlation coefficient= %f" % SRCC[0])
    KTAU = scipy.stats.kendalltau(sys_true, sys_predicted)
    print("[SYSTEM] Kendall Tau rank correlation coefficient= %f" % KTAU[0])

    ## generate answer.txt for codalab
    if not os.path.dirname(outfile) == "":
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
    ans = open(outfile, "w")
    for k, v in predictions.items():
        outl = k.split(".")[0] + "," + str(v) + "\n"
        ans.write(outl)
    ans.close()


if __name__ == "__main__":
    main()
