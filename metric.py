# Reference
# https://github.com/ShiyuLiang/odin-pytorch
# Thanks for @ShiyuLiang

# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import numpy as np

from sklearn.metrics import roc_curve, auc, precision_recall_curve

def auroc(normal_score, anomal_score):
    truth = np.concatenate((np.zeros_like(anomal_score), np.ones_like(normal_score)))
    preds = np.concatenate((anomal_score, normal_score))
    fpr, tpr, roc_thresholds = roc_curve(truth, preds)
    aurocBase = auc(fpr, tpr)

    return aurocBase


def auprIn(normal_score, anomal_score):
    truth = np.concatenate((np.zeros_like(anomal_score), np.ones_like(normal_score)))
    preds = np.concatenate((anomal_score, normal_score))
    precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(truth, preds)
    auprBase = auc(recall_norm, precision_norm)

    return auprBase

def auprOut(normal_score, anomal_score):
    truth = np.concatenate((np.zeros_like(anomal_score), np.ones_like(normal_score)))
    preds = np.concatenate((anomal_score, normal_score))
    # pr curve where "anomaly" is the positive class
    precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(truth, -preds, pos_label=0)
    auprBase = auc(recall_anom, precision_anom)

    return auprBase

def detection(normal_score, anomal_score, start, end, gap):
    # calculate the minimum detection error
    errorBase = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(normal_score < delta)) / np.float(len(normal_score))
        error2 = np.sum(np.sum(anomal_score > delta)) / np.float(len(anomal_score))
        errorBase = np.minimum(errorBase, (tpr + error2) / 2.0)

    return errorBase

def metric(normal_score, anomal_score):
    start = np.min([np.min(normal_score), np.min(anomal_score)])
    end = np.max([np.max(normal_score), np.max(anomal_score)])
    gap = (end - start) / 100000

    errorBase = detection(normal_score, anomal_score, start, end, gap)
    aurocBase = auroc(normal_score, anomal_score)
    auprinBase = auprIn(normal_score, anomal_score)
    auproutBase = auprOut(normal_score, anomal_score)

    metric_dic = dict()
    metric_dic['Detection-error'] = errorBase
    metric_dic['AUROC'] = aurocBase
    metric_dic['AUPR-In'] = auprinBase
    metric_dic['AUPR-Out'] = auproutBase

    return metric_dic