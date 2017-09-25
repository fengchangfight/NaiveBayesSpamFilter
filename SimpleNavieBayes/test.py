#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

"""
@Author: MarkLiu
"""
import SimpleNavieBayes.NavieBayes as naiveBayes
import random
import numpy as np


def simpleTest():
    # load saved model from training
    vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam = \
        naiveBayes.getTrainedModelInfo()

    # load test data
    filename = '../emails/test/test.txt'
    smsWords, classLables = naiveBayes.loadSMSData(filename)

    smsType = naiveBayes.classify(vocabularyList, pWordsSpamicity,
                                  pWordsHealthy, pSpam, smsWords[0])
    print smsType


def testClassifyErrorRate():
    """
    error rate test
    :return:
    """
    filename = '../emails/training/SMSCollection.txt'
    smsWords, classLables = naiveBayes.loadSMSData(filename)

    # cross validation
    testWords = []
    testWordsType = []

    testCount = 1000
    for i in range(testCount):
        randomIndex = int(random.uniform(0, len(smsWords)))
        testWordsType.append(classLables[randomIndex])
        testWords.append(smsWords[randomIndex])
        del (smsWords[randomIndex])
        del (classLables[randomIndex])

    vocabularyList = naiveBayes.createVocabularyList(smsWords)
    print "generate one hot vector based on the word set！"
    trainMarkedWords = naiveBayes.setOfWordsListToVecTor(vocabularyList, smsWords)
    print "mark data！"
    # convert to nd array
    trainMarkedWords = np.array(trainMarkedWords)
    print "data -> matrix！"
    pWordsSpamicity, pWordsHealthy, pSpam = naiveBayes.trainingNaiveBayes(trainMarkedWords, classLables)

    errorCount = 0.0
    for i in range(testCount):
        smsType = naiveBayes.classify(vocabularyList, pWordsSpamicity,
                                      pWordsHealthy, pSpam, testWords[i])
        print 'predict type：', smsType, 'actual type：', testWordsType[i]
        if smsType != testWordsType[i]:
            errorCount += 1

    print 'error count：', errorCount, 'error rate：', errorCount / testCount


if __name__ == '__main__':
    testClassifyErrorRate()
