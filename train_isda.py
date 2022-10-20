import json
import os
import pickle
import time
from os.path import join

import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from ISDA import ISDALoss,EstimatorCV


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train_isda(model,fc, train_loader, eval_loader, num_epochs, output, eval_each_epoch,lambda_0):
    utils.create_dir(output)
    optim = torch.optim.Adamax([{'params': model.parameters()},{'params': fc.parameters()}])
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    #need to rewrite the isda_loss
    isda_criterion=ISDALoss(int(model.feature_num),int(model.class_num)).cuda()
    ce_criterion = nn.CrossEntropyLoss().cuda()
    all_results = []

    total_step = 0

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0

        t = time.time()
        
        ratio =lambda_0 * (epoch / num_epochs)

        model.train()
        fc.train()

        for i, (v, q, a, b) in tqdm(enumerate(train_loader), ncols=100,
                                    desc="Epoch %d" % (epoch+1), total=len(train_loader)):
            total_step += 1
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            b = Variable(b).cuda()

            loss, pred = isda_criterion(model, fc, v,q,a,b, ratio)

            if (loss != loss).any():
              raise ValueError("NaN loss")
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            nn.utils.clip_grad_norm(fc.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data[0] * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)

        run_eval = eval_each_epoch or (epoch == num_epochs - 1)

        if run_eval:
            model.train(False)
            fc.train(False)
            results = evaluate(model,fc, eval_loader)
            results["epoch"] = epoch+1
            results["step"] = total_step
            results["train_loss"] = total_loss
            results["train_score"] = train_score
            data_r=list(results)
            all_results.append(data_r)

            with open(join(output, "results.json"), "w") as f:
                json.dump(all_results, f, indent=2)

            eval_score = results["score"]
            bound = results["upper_bound"]

        logger.write('epoch %d, time: %.2f' % (epoch+1, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))

        if run_eval:
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

    model_path = os.path.join(output, 'model.pth')
    torch.save(model.state_dict(), model_path)
    fc_path = os.path.join(output, 'fc.pth')
    torch.save(fc.state_dict(), fc_path)



def evaluate(model,fc, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0

    all_logits = []
    all_bias = []
    for v, q, a, b in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        with torch.no_grad():
            v = Variable(v).cuda()
            q = Variable(q).cuda()
        feature = model(v, None, q, None, None)
        pred=fc(feature)
        all_logits.append(pred.data.cpu().numpy())

        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)
        all_bias.append(b)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    results = dict(
        score=score,
        upper_bound=upper_bound,
    )
    return results