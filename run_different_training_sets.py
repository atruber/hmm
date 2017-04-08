# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 14:17:49 2016

@author: adrianne
"""
from corpus import Document, NPChunkCorpus
from hmm import HMM
from unittest import TestCase, main
from evaluator import compute_cm
from random import shuffle, seed
import sys, copy


class HMMTest(TestCase):
    u"""Tests for the HMM sequence labeler."""
    
    def run_all(self):
        print '-----FEATURES = WORDS-----'

    def split_np_chunk_corpus(self, document_class):
        """Split the yelp review corpus into training, dev, and test sets"""
        sentences = NPChunkCorpus('np_chunking_wsj', document_class=document_class)
        seed(hash("np_chunk"))
        shuffle(sentences)
        return (sentences[:8936], sentences[8936:])

    def test_sets(self):
        """Test NP chunking with word and postag feature"""
        train, test = self.split_np_chunk_corpus(Document)
        train1,train2,test1,test2= copy.deepcopy(train), copy.deepcopy(train), copy.deepcopy(test), copy.deepcopy(test)
        
        for i in range(len(train)):
            for j in range(len(train[i].data)):
                train1[i].data[j] = train[i].data[j][0]
                train2[i].data[j] = train[i].data[j][1]
        for l in range(len(test)):
            for k in range(len(test[l].data)):
                test1[l].data[k] = test[l].data[k][0]
                test2[l].data[k] = test[l].data[k][1]
        
        classifier = HMM()
        
        print "\n---------------FEATURES = WORDS---------------"
        classifier.train(train1)
        test_result = compute_cm(classifier, test1)
        _, _, f1, accuracy = test_result.print_out()
        self.assertGreater(accuracy, 0.55)
        self.assertTrue(all(i>=.90 for i in f1), 'not all greater than 90.0%')
        
        print "\n---------------FEATURES = POS---------------"
        classifier.train(train2)
        test_result = compute_cm(classifier, test2)
        _, _, f1, accuracy = test_result.print_out()
        self.assertGreater(accuracy, 0.55)
        self.assertTrue(all(i>=.90 for i in f1), 'not all greater than 90.0%')
        
        print "\n---------------FEATURES = WORDS+POS---------------"
        classifier.train(train)
        test_result = compute_cm(classifier, test)
        _, _, f1, accuracy = test_result.print_out()
        self.assertGreater(accuracy, 0.55)
        self.assertTrue(all(i>=.90 for i in f1), 'not all greater than 90.0%')
    

if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main(verbosity=2)
    