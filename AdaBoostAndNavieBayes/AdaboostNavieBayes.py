#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

import numpy as np

# ==fc==
def textParser(text):
    """
    preprocessing，remove empty string, to lower case
    :param text:
    :return:
    """
    import re
    regEx = re.compile(r'[^a-zA-Z]|\d')  # remove everything except for words and numbers using regular expression
    words = regEx.split(text)

    words = [word.lower() for word in words if len(word) > 0]
    return words

# ==fc==
def loadSMSData(fileName):
    """
    load SMS data from file name
    :param fileName:
    :return:
    """
    f = open(fileName)
    classCategory = []  # 类别标签，1表示是垃圾SMS，0表示正常SMS
    smsWords = []
    for line in f.readlines():
        linedatas = line.strip().split('\t')
        if linedatas[0] == 'ham':
            classCategory.append(0)
        elif linedatas[0] == 'spam':
            classCategory.append(1)

        words = textParser(linedatas[1])
        smsWords.append(words)
    return smsWords, classCategory

# ==fc==
def createVocabularyList(smsWords):
    """
    create vector of word set
    :param smsWords:
    :return:
    """
    vocabularySet = set([])
    for words in smsWords:
        vocabularySet = vocabularySet | set(words)
    vocabularyList = list(vocabularySet)
    return vocabularyList

# ==fc==
def getVocabularyList(fileName):
    """
    get vocabulary list from file
    :param fileName:
    :return:
    """
    fr = open(fileName)
    vocabularyList = fr.readline().strip().split('\t')
    fr.close()
    return vocabularyList

# ==fc==
def setOfWordsToVecTor(vocabularyList, smsWords):
    """
    mark word occurence in a way like this vector: [0,0,1,5,0,3,4...]
    :param vocabularyList:
    :param smsWords:
    :return:
    """
    vocabMarked = [0] * len(vocabularyList)
    for smsWord in smsWords:
        if smsWord in vocabularyList:
            vocabMarked[vocabularyList.index(smsWord)] += 1
    return np.array(vocabMarked)

# ==fc==
def setOfWordsListToVecTor(vocabularyList, smsWordsList):
    """
    stack setOfWordsToVecTor, return a 2d array
    :param vocabularyList:
    :param smsWordsList:
    :return:
    """
    vocabMarkedList = []
    for i in range(len(smsWordsList)):
        vocabMarked = setOfWordsToVecTor(vocabularyList, smsWordsList[i])
        vocabMarkedList.append(vocabMarked)
    return vocabMarkedList

# ==fc==
def trainingNaiveBayes(trainMarkedWords, trainCategory):
    """
    calculate spamicity：P（Wi|S）
    :param trainMarkedWords: marked 2d array data
    :param trainCategory:
    :return:
    """
    numTrainDoc = len(trainMarkedWords)
    numWords = len(trainMarkedWords[0])
    # prior probability of spam P(S)
    pSpam = sum(trainCategory) / float(numTrainDoc)

    wordsInSpamNum = np.ones(numWords)
    wordsInHealthNum = np.ones(numWords)
    spamWordsNum = 2.0
    healthWordsNum = 2.0
    for i in range(0, numTrainDoc):
        if trainCategory[i] == 1:  # if spam
            wordsInSpamNum += trainMarkedWords[i]
            spamWordsNum += sum(trainMarkedWords[i])  #  total spam occurance
        else:
            wordsInHealthNum += trainMarkedWords[i]
            healthWordsNum += sum(trainMarkedWords[i])

    pWordsSpamicity = np.log(wordsInSpamNum / spamWordsNum)
    pWordsHealthy = np.log(wordsInHealthNum / healthWordsNum)

    return pWordsSpamicity, pWordsHealthy, pSpam

# ==fc==
def getTrainedModelInfo():
    """
    get training model
    :return:
    """
    # load training info from training
    vocabularyList = getVocabularyList('vocabularyList.txt')
    pWordsHealthy = np.loadtxt('pWordsHealthy.txt', delimiter='\t')
    pWordsSpamicity = np.loadtxt('pWordsSpamicity.txt', delimiter='\t')
    fr = open('pSpam.txt')
    pSpam = float(fr.readline().strip())
    fr.close()

    return vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam


def classify(pWordsSpamicity, pWordsHealthy, DS, pSpam, testWordsMarkedArray):
    """
    calculate joint probability
    :param testWordsMarkedArray:
    :param pWordsSpamicity:
    :param pWordsHealthy:
    :param DS:  adaboost weight!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! here's the difference from the normal version
    :param pSpam:
    :return:
    """
    # to calculate P(Ci|W)，W is vector。P(Ci|W) only need to calculate P(W|Ci)P(Ci)
    ps = sum(testWordsMarkedArray * pWordsSpamicity * DS) + np.log(pSpam)
    ph = sum(testWordsMarkedArray * pWordsHealthy) + np.log(1 - pSpam)
    if ps > ph:
        return ps, ph, 1
    else:
        return ps, ph, 0
