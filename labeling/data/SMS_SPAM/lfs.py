from labeling.lf import *
from labeling.preprocess import *
from labeling.continuous_scoring import *
from labeling.lf_set import *

from con_scorer import word_similarity

import numpy as np
import re

SPAM = 1
HAM = 0
ABSTAIN = -1


trigWord1 = {"free","credit","cheap","apply","buy","attention","shop","sex","soon","now","spam"}
trigWord2 = {"gift","click","new","online","discount","earn","miss","hesitate","exclusive","urgent"}
trigWord3 = {"cash","refund","insurance","money","guaranteed","save","win","teen","weight","hair"}
notFreeWords = {"toll","Toll","freely","call","meet","talk","feedback"}
notFreeSubstring = {"not free","you are","when","wen"}
firstAndSecondPersonWords = {"I","i","u","you","ur","your","our","we","us","youre"}
thirdPersonWords = {"He","he","She","she","they","They","Them","them","their","Their"}


@labeling_function(resources=dict(keywords=trigWord1))
def LF1(c,keywords):
    if len(keywords.intersection(c.split())) > 0:
        return SPAM
    else:
        return ABSTAIN


@labeling_function(resources=dict(keywords=trigWord2))
def LF2(c,keywords):
    if len(keywords.intersection(c.split())) > 0:
        return SPAM
    else:
        return ABSTAIN

@labeling_function(resources=dict(keywords=trigWord3))
def LF3(c,keywords):
    if len(keywords.intersection(c.split())) > 0:
        return SPAM
    else:
        return ABSTAIN

@labeling_function(resources=dict(keywords=notFreeWords))
def LF4(c,keywords):
    if "free" in c.split() and len(keywords.intersection(c.split()))>0:
        return HAM
    else:
        return ABSTAIN

@labeling_function(resources=dict(keywords=notFreeSubstring))
def LF5(c,keywords):
    for pattern in keywords:    
        if "free" in c.split() and re.search(pattern,c, flags= re.I):
            return HAM
    return ABSTAIN

@labeling_function(resources=dict(keywords=firstAndSecondPersonWords))
def LF6(c,keywords):
    if "free" in c.split() and len(keywords.intersection(c.split()))>0:
        return HAM
    else:
        return ABSTAIN


@labeling_function(resources=dict(keywords=thirdPersonWords))
def LF7(c,keywords):
    if "free" in c.split() and len(keywords.intersection(c.split()))>0:
        return HAM
    else:
        return ABSTAIN

@labeling_function()
def LF8(c):
    if (sum(1 for ch in c if ch.isupper()) > 6):
        return SPAM
    else:
        return ABSTAIN

@labeling_function()
def LF9(c):
    return HAM

@labeling_function(cont_scorer=word_similarity,resources=dict(keywords=trigWord1))
def CLF1(c,keywords):
    return SPAM

@labeling_function(cont_scorer=word_similarity,resources=dict(keywords=trigWord2))
def CLF2(c,keywords):
    return SPAM

@labeling_function(cont_scorer=word_similarity,resources=dict(keywords=trigWord3))
def CLF3(c,keywords):
    return SPAM

@labeling_function(cont_scorer=word_similarity,resources=dict(keywords=notFreeWords))
def CLF4(c,keywords):
    return HAM

@labeling_function(cont_scorer=word_similarity,resources=dict(keywords=notFreeSubstring))
def CLF5(c,keywords):
    return HAM

@labeling_function(cont_scorer=word_similarity,resources=dict(keywords=firstAndSecondPersonWords))
def CLF6(c,keywords):
    return HAM

@labeling_function(cont_scorer=word_similarity,resources=dict(keywords=thirdPersonWords))
def CLF7(c,keywords):
    return HAM


LFS = [LF1,
    LF2,
    LF3,
    LF4,
    LF5,
    LF6,
    LF7,
    LF8,
    LF9,
    CLF1,
    CLF2,
    CLF3,
    CLF4,
    CLF5,
    CLF6,
    CLF7]

rules = LFSet("set1")
rules.add_lf_list(LFS)
