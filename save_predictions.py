import argparse
import json
from os import listdir
from os.path import join, exists, isdir
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict
import base_model
from dataset import VQAFeatureDataset, Dictionary


def main():
    parser = argparse.ArgumentParser("Save a model's predictions for the VQA-CP test set")
    parser.add_argument("model", help="Directory of the model")
    parser.add_argument("output_file", help="File to write json output to")
    args = parser.parse_args()

    path = args.model

    print("Loading data...")
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary, cp=False)
    eval_dset = VQAFeatureDataset('val', dictionary, cp=False)

    eval_loader = DataLoader(eval_dset, 256, shuffle=False, num_workers=0)

    constructor = 'build_%s' % 'baseline0_isda'
    model = getattr(base_model, constructor)(train_dset, 1024).cuda()
    fc = base_model.Full_layer(int(model.feature_num), int(model.class_num)).cuda()
    print("Loading state dict for %s..." % path)

    state_dict_model = torch.load(join(path, "model.pth"))
    state_dict_fc=torch.load(join(path,"fc.pth"))
    state_dict = torch.load(join(path, "model.pth"))
    if all(k.startswith("module.") for k in state_dict):
        filtered = {}
        for k in state_dict:
            filtered[k[len("module."):]] = state_dict[k]
        state_dict = filtered

    for k in list(state_dict):
        if k.startswith("debias_loss_fn"):
            del state_dict[k]
    model.load_state_dict(state_dict_model)
    fc.load_state_dict(state_dict_fc)
    model.cuda()
    model.eval()
    fc.cuda()
    fc.eval()
    
    print("Done")

    predictions = []
    for v, q, a, b in tqdm(eval_loader, ncols=100, total=len(eval_loader), desc="eval"):
        with torch.no_grad():
            v = Variable(v).cuda()
            q = Variable(q).cuda()
        factor = model(v, None, q, None, None, True)
        print(factor.shape)
        pred=fc(factor)
        print(pred.shape)
        prediction = torch.max(pred, 1)[1].data.cpu().numpy()
        for p in prediction:
            predictions.append(train_dset.label2ans[p])

    out = []
    for p, e in zip(predictions, eval_dset.entries):
        out.append(dict(answer=p, question_id=e["question_id"]))
    with open(join(path, args.output_file), "w") as f:
        json.dump(out, f)


if __name__ == '__main__':
    main()
