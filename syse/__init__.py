import pandas as pd
from stat_parser import Parser
import re
import numpy as np
import unicodedata
import math
import os
import sys

#Syntactic Sentence Extraction (SySE)

#What is sentence extraction? 
#Wikipedia has a pretty good description:
#https://en.wikipedia.org/wiki/Sentence_extraction
#If you're offline, it is simply an automatic way to summarize text, by \
#extracing what are thought to be the most important sentences.

#How is this program different from others?
#This program attempts to summarize an article by looking at the syntax of \
#its phrases; it doesn't look at all what words are being used, just at what \
#kinds of words, and the word's dependencies on one another.
class SySE:
    def __init__(self):
        self.loadParameters(os.path.dirname(os.path.realpath(sys.argv[0])) + 'default')
    ####Supervised Training.
    #trainingSentences: sentences on which to train (Must already be parsed)
    #labels: corresponding binary (1,0) labels.
    #alpha: laplace/additive smoothing parameter (default = 1)
    def train(self, trainingSentences, labels, alpha = 0.1, beta = 0.1, debug = 0):
        if debug > -1:
            print
            print '*********************************************************'
            print '                   SySE V 0.1 '
            print 'Beginning Training Sequence with ' + \
                str(len(trainingSentences)) + ' training sentences...'
            print '*********************************************************'
            if debug > 0:
                print
                print 'Initializing... '
        
        if type(trainingSentences[0]) != list:
            print 'These sentences do not appear to have been parsed.'
            print 'They will be parsed now.'
            if len(trainingSentences > 10):
                print 'Given their volume, this will take some time.'
            try:
                self.parser = Parser()
            except:
                print 'This environment should have pystatparser loaded ' + \
                'in order to train on unparsed sentences.'
                print 'Exiting...'
                return
            trainingSentences = [parser.parse(x) for x in trainingSentences]
        
        ####Initialization
        #Save hyperparameters
        self.alpha = alpha
        self.beta = beta
        
        #See what tags are in the training data.
        tags = []
        for sentence in trainingSentences:
            flat = self.recursiveFlatten(sentence)
            for el in flat:
                if type(el) == unicode and el not in tags:
                    tags.append(el)            
            
        self.tags = set(tags)
        
        #What kind of root tags are there?
        self.sentenceTypes = set([x[0] for x in trainingSentences])
        
        #Which tags may contain other tags?
        self.phraseTags = []
        for sentence in trainingSentences:
            flat = self.recursiveFlatten(sentence)
            for i in range(0,len(flat)):
                try:
                    if type(flat[i]) == unicode and type(flat[i+1]) == unicode and flat[i] not in self.phraseTags:
                        self.phraseTags.append(flat[i])
                except IndexError:
                    print 'We\'ve reached the end of this sentence'
                    
        self.phraseTags = set(self.phraseTags) - self.sentenceTypes
       
        #Robustness
        labels = list(labels)    
        
        #Split training sentences into Important (I) and Regular (R) (Unimportant)
        importantSentences = filter(lambda x: labels[trainingSentences.index(x)]==1, trainingSentences)
        regularSentences = filter(lambda x: not labels[trainingSentences.index(x)]==1, trainingSentences)
    
        self.classPriors = []
        
        ###Test inputs
        #Make sure labels are right length for sentences.
        if len(labels) != len(trainingSentences):
            print 'Labels and trainingSentencs must be the same length!'
            return
        #Make sure labels are valid
        for label in labels:
            if label != 0 and label != 1:
                print 'Lables should be either 0 or 1.'
                print 'exiting...'
                return
                
        #REMOVE THIS BEFORE PUBLISHING
        #Split training sentences into Important (I) and Regular (R) (Unimportant)
        #self.importantRootProbabilities = filter(lambda x: labels[trainingSentences.index(x)]==1, trainingSentences)
        #self.regularRootProbabilities = filter(lambda x: not labels[trainingSentences.index(x)]==1, trainingSentences)
                            
        ###Train Class Priors
        self.classPriors.append(float(labels.count(0))/float(len(labels)))
        self.classPriors.append(float(labels.count(1))/float(len(labels)))
        
        if debug > 0:
            print '*********************************************************'
            print 'These are the class priors'
            print '*********************************************************'    
            print self.classPriors
            print
            print      
        
        ###Train Sentence Type
        self.importantRootProbabilities = dict(zip(list(self.sentenceTypes),np.zeros(len(list(self.sentenceTypes)))))
        self.regularRootProbabilities = dict(zip(list(self.sentenceTypes),np.zeros(len(list(self.sentenceTypes)))))
        
        #Get the count of each sentence type in I
        for sentence in importantSentences:
            #Make sure we get what we expect
            if type(sentence[0]) != unicode:
                print "We are looking for a non-unicode sentence type. exiting..."
                break
                return            
            #if it isn't in the list yet, add it.
            self.importantRootProbabilities[sentence[0]] += 1
            
        #We will now implement a softmax to turn the counts into probabilities
        for param in self.importantRootProbabilities:
            self.importantRootProbabilities[param]=(float(self.importantRootProbabilities[param]) + alpha)/ \
                (float(len(trainingSentences)) + alpha*(len(self.importantRootProbabilities)+1))
                
        #Get the count of each sentence type in R
        for sentence in regularSentences:
            #Make sure we get what we expect
            if type(sentence[0]) != unicode:
                print "We are looking for a non-unicode sentence type. exiting..."
                break
                return            
            #if it isn't in the list yet, add it.
            self.regularRootProbabilities[sentence[0]] += 1
        
        #We will now implement a softmax to turn the counts into probabilities
        for param in self.regularRootProbabilities:
            self.regularRootProbabilities[param]=float(self.regularRootProbabilities[param])/ \
                float(len(trainingSentences))
        
        if debug > 0:
            print '*********************************************************'
            print 'These are the sentence type parameters'
            print '*********************************************************'
            
            print ' --------------------------------------------------------'
            print ' For Important Sentences:'
            print self.importantRootProbabilities
            
            print ' --------------------------------------------------------'
            print ' For Regular Sentences:'
            print self.regularRootProbabilities
            print
            print
        
        ###Train Phrases
        
        ##Primitive Inference on Multiplicity Parameter
        #Define dictionaries to store times a tag was included in a phrase
        tagInclusionI = dict(zip(list(self.tags), np.zeros(len(list(self.tags)))))#How many times is a tag in a level?
        tagInclusionR = dict(zip(list(self.tags), np.zeros(len(list(self.tags)))))#How many times is a tag in a level?
        #Define dictionaries to store times a tag was used at all.
        tagCountI = dict(zip(list(self.tags), np.zeros(len(list(self.tags)))))#How many total times does the tag appear?
        tagCountR = dict(zip(list(self.tags), np.zeros(len(list(self.tags)))))#How many total times does the tag appear?
        #To store dumb poisson inference
        self.importantMultiplictyParameter = dict(zip(list(self.tags), np.zeros(len(list(self.tags)))))#For storing parameter estimates.
        self.regularMultiplictyParameter = dict(zip(list(self.tags), np.zeros(len(list(self.tags)))))#For storing parameter estimates.
        
        #Get Inclusion for I
        for sentence in importantSentences:
            self.getInclusions([sentence,tagInclusionI,debug>=2])
        
        #Get Inclusion for R
        for sentence in regularSentences:
            self.getInclusions([sentence,tagInclusionR,debug>=2])
        
        #Get Counts for I
        for sentence in importantSentences:
            flat = self.recursiveFlatten(sentence)
            currentTags = filter(lambda x: type(x)==unicode, flat)
            for tag in currentTags[1:]:
                tagCountI[tag] += 1
        
        #Get Counts for R
        for sentence in regularSentences:
            flat = self.recursiveFlatten(sentence)
            currentTags = filter(lambda x: type(x)==unicode, flat)
            for tag in currentTags[1:]:
                tagCountR[tag] += 1
        
        #Estimate Parameters for I
        for tag in tagInclusionI.keys():
            if (tagCountI[tag] > 1):
                self.importantMultiplictyParameter[tag] = (tagCountI[tag]-1) / tagInclusionI[tag]
                
        #Estimate Parameters for R
        for tag in tagInclusionR.keys():
            if (tagCountR[tag] > 1):
                self.regularMultiplictyParameter[tag] = (tagCountR[tag]-1) / tagInclusionR[tag]
        
            if debug > 0:
                print '*********************************************************'
                print ' Estimation for Multiplicity Parameters '
                print '*********************************************************'
                print
                print ' ------------------------------------------------------------------'
                print 'Tag Counts for Important Sentences:'
                print tagCountI    
                print 'Tag Counts for Regular Sentences:'
                print tagCountR
                print ' ------------------------------------------------------------------'
                print 'Tag Inclusion for Important Sentences:'
                print tagInclusionI
                print 'Tag Inclusion for Regular Sentences:'
                print tagInclusionR
                print ' ------------------------------------------------------------------'
                print 'Dumb Parameter Estimates for Imporant Sentences:'
                print self.importantMultiplictyParameter
                print 'Dumb Parameter Estimates for Regular Sentences:'
                print self.regularMultiplictyParameter    
                print
                print
        
        ##Primitive Inference on Presence Parameters
        
        #We need to find inclusions given parent
        #To store conditional presence probabilities, what can almost be \
            #thought of as transition probabilities.
        #This is the uinformed probability of a particular presence.
        ui = self.alpha / (self.alpha*(len(self.regularRootProbabilities) + 1))
        #For important phrases
        self.importantCondPresenceProbs = np.zeros([len(self.tags),len(self.phraseTags) + len(self.sentenceTypes)])
        self.importantCondPresenceProbs = pd.DataFrame(self.importantCondPresenceProbs).applymap(lambda x: x + ui)
        self.importantCondPresenceProbs.columns = list(self.sentenceTypes) + list(self.phraseTags)
        self.importantCondPresenceProbs.index = list(self.tags)
        
        #For regularPhrases
        self.regularCondPresenceProbs = np.zeros([len(self.tags),len(self.phraseTags) + len(self.sentenceTypes)])
        self.regularCondPresenceProbs = pd.DataFrame(self.regularCondPresenceProbs).applymap(lambda x: x + ui)
        self.regularCondPresenceProbs.columns = list(self.sentenceTypes) + list(self.phraseTags)
        self.regularCondPresenceProbs.index = list(self.tags)
        
        #Define dictionaries to store times a tag was used at all. This time, \
            #We care about root/sentence tags as well.
        tagCountI = dict(zip(list(self.tags), np.zeros(len(list(self.tags)))))#How many total times does the tag appear?
        tagCountR = dict(zip(list(self.tags), np.zeros(len(list(self.tags)))))#How many total times does the tag appear?
        
        #Tag counts, but on sentences as well, unlike above.
        
        
        #Count Conditional Inclusions for Important Sentences
        for sentence in importantSentences:
            self.getInclusionsGivenParent([sentence,self.importantCondPresenceProbs,sentence[0],debug>=2])
        
        #Count Conditional Inclusions for Regular Sentences
        for sentence in regularSentences:
            self.getInclusionsGivenParent([sentence,self.regularCondPresenceProbs,sentence[0],debug>=2])
        
        #Get Counts for I
        for sentence in importantSentences:
            flat = self.recursiveFlatten(sentence)
            currentTags = filter(lambda x: type(x)==unicode, flat)
            for tag in currentTags:
                tagCountI[tag] += 1
        
        #Get Counts for R
        for sentence in regularSentences:
            flat = self.recursiveFlatten(sentence)
            currentTags = filter(lambda x: type(x)==unicode, flat)
            for tag in currentTags:
                tagCountR[tag] += 1
        
        #Calculate Conditional Presence Parameter for Important Sentences
        for column in self.importantCondPresenceProbs.columns:
            if tagCountI[column] > 0:
                num = self.importantCondPresenceProbs.loc[:,column] + alpha
                denom = tagCountI[column] + (len(self.importantCondPresenceProbs.columns) + 1)*alpha
                self.importantCondPresenceProbs.loc[:,column] = num/denom
            
        #Calculate Conditional Presence Parameter for Regular Sentences
        for column in self.regularCondPresenceProbs.columns:
            if tagCountR[column] > 0:
                #AdditiveSmoothing
                num = self.regularCondPresenceProbs.loc[:,column] + alpha
                denom = tagCountR[column] + (len(self.regularCondPresenceProbs.columns) + 1)*alpha
                self.regularCondPresenceProbs.loc[:,column] = num/denom
        
        if debug > 1:
            print '*********************************************************'
            print 'Presence Parameter Estimation'
            print '*********************************************************'
            print
            print ' ------------------------------------------------------------------'
            print ' Conditional Parameters for Important Sentences'
            print self.importantCondPresenceProbs
            print
            print ' ------------------------------------------------------------------'
            print ' Conditional Parameters for Regular Sentences'
            print self.regularCondPresenceProbs
            print ' ------------------------------------------------------------------'    
        
        if debug > -1:
            print
            print
            print '...Finished'
        
        ####Classification
    def classify(self, sentence, debug = 0):
        #If the sentence hasn't been parsed, we must parse it.
        plaintext = False
        if type(sentence) != list:
            plaintext = True
            original = sentence
            try:
                sentence = self.parser.parse(sentence)
            except:
                try:
                    self.parser = Parser()
                    sentence = self.parser.parse(sentence)
                except:
                    print 'Couldn\'t create a parsing object.'
                    print 'Prehaps pystatparser is not loaded?'
                    print 'type \"from stat_parser import Parser\"'
            
        #Deal with new root types
        if sentence[0] not in self.importantRootProbabilities:
            self.importantRootProbabilities[sentence[0]] = self.alpha/(self.alpha*(len(self.importantRootProbabilities) + 1))
        if sentence[0] not in self.regularRootProbabilities:
            self.regularRootProbabilities[sentence[0]] = self.alpha/(self.alpha*(len(self.regularRootProbabilities) + 1))
        
        #Deal with new non-root tag types
        flat = self.recursiveFlatten(sentence)
        flat = filter(lambda x: type(x) == unicode, flat)
        for i,tag in enumerate(flat):
            if tag not in self.tags:
                #Set a priori beliefs for multiplicity parameters
                self.importantMultiplictyParameter[tag] = self.beta
                self.regularMultiplictyParameter[tag] = self.beta
                
                #Set a priori beliefs for conditional presence parameters
                self.importantCondPresenceProbs.loc[tag] = np.repeat(self.alpha / (self.alpha*(len(self.regularRootProbabilities) + 1)),len(self.importantCondPresenceProbs.columns))
                self.regularCondPresenceProbs.loc[tag] = np.repeat(self.alpha / (self.alpha*(len(self.regularRootProbabilities) + 1)),len(self.regularCondPresenceProbs.columns))
                if type(flat[i+1])==unicode:
                    #Set a priori beliefs for conditional presence parameters
                    self.importantCondPresenceProbs[tag] = np.repeat(self.alpha / (self.alpha*(len(self.regularRootProbabilities) + 1)),len(self.importantCondPresenceProbs.index))
                    self.regularCondPresenceProbs[tag] = np.repeat(self.alpha / (self.alpha*(len(self.regularRootProbabilities) + 1)),len(self.regularCondPresenceProbs.index))
        
        ##Get P(x|y = REGULAR) 
        PxGy1 = math.log(self.importantRootProbabilities[sentence[0]]) + self.getConditionalLevelProbability([sentence,self.importantCondPresenceProbs,self.importantMultiplictyParameter,sentence[0],debug>=2])
        
        ##Get P(x|y = REGULAR) 
        PxGy0 = math.log(self.regularRootProbabilities[sentence[0]]) + self.getConditionalLevelProbability([sentence,self.regularCondPresenceProbs,self.regularMultiplictyParameter,sentence[0],debug>=2])
        
        #Get priors in a log form:
        Py1 = math.log(self.classPriors[1])
        Py0 = math.log(self.classPriors[0])
        
        #Get log Probabilities of each class through Bayes' Rule
        Py1Gx = PxGy1+Py1
        Py0Gx = PxGy0+Py0
        
        #SoftMax probabilities
        denom = math.log(math.e**Py1Gx + math.e**Py0Gx)
        
        sPy1Gx = Py1Gx-denom
        sPy0Gx = Py0Gx-denom
        
        #Turn back into probabilities for output
        sPy1Gx = math.e**sPy1Gx
        sPy0Gx = math.e**sPy0Gx
        
        if debug > -1:
            print 'Estimating Class for sentence:'
            if plaintext:
                print '\"' + original + '\"'
            else:
                print sentence
        if debug > 0:
            print ' ------------------------------------------------------------------'
            print 'Class Priors (log probability):'
            print 'P(important) = ' + str(Py1)
            print 'P(unimportant) = ' + str(Py0)
            print ' ------------------------------------------------------------------'
            print 'Conditional Sentence Log Probabilities:'
            print 'P(sentence | important) = ' + str(PxGy1)
            print 'P(sentence | unimportant) = ' + str(PxGy0)
            print ' ------------------------------------------------------------------'
            print 'Unnormalized Conditional Class Log Probabilities'
            print 'P(important | sentence) = ' + str(Py1Gx)
            print 'P(unimportant | sentence) = ' + str(Py0Gx)
        if debug > -1:
            print ' ------------------------------------------------------------------'
            print 'Softmaxed Conditional Class Probabilities'
            print 'P(important | sentence) = ' + str(sPy1Gx)
            print 'P(unimportant | sentence) = ' + str(sPy0Gx)
        return(sPy1Gx)
    
    def summarize(self, article, verbosity = 0.5, debug = 0):
        sentences = self.split_into_sentences(article)
        
        keepers = []        
        i = 0
        for sentence in sentences:
            i += 1
            try:
                if self.classify(sentence, debug = debug) > verbosity:
                    keepers.append(sentence)
            except:
                print 'Error classifying sentence ' + str(i)
                print 'FullText: ' 
                print sentence
        if len(keepers) == 0:
            print 'No sentences found important'
            return('')
        reduced = reduce(lambda x,y: x + ' ' + y, keepers)
        return(reduced)
    
    ####Function Definitions
    #Returns the log probability of a level occuring, along with using recursion to \
    #find the levels contained therein. May be passed an entire sentence.
    def getConditionalLevelProbability(self, inputs):
        level = inputs[0]
        tagDF = inputs[1]
        mult = inputs[2]
        parent = inputs[3]
        debug = inputs[4]
        ret = 0
        if debug == 1:
            print 'Beginning Level...........'
            print level
        inTags = [x[0] for x in level[1:]]
        if u'' in inTags:
            inTags.remove(u'')
        
        #Do some recursion
        for i,tag in enumerate(inTags):
            if tag in self.phraseTags or tag in self.sentenceTypes:
                if debug == 1:
                    print 'beginning recursion due to:'
                    print tag
                ret = ret + self.getConditionalLevelProbability([level[i+1],tagDF,mult,tag,debug])
        
        #Do multiplicity for this level
        for tag in inTags:
            x = inTags.count(tag)
            mu = mult[tag]
            ret = ret + 0#math.log((math.exp(-mu) * mu**x / math.factorial(x)))
        
        #Do presence for this level
        inTags = list(set(inTags))
        for tag in inTags:
            if type(tag) != unicode:#Some sentences contain only a word, and we won't need to add anything in that case.
                print 'Breaking Due to non-unicode tag in getConditionalLevelProbability!'
                print tag
                break
            if debug == 1:
                print 'Probability of ' + tag + ' given ' + parent + ' is ' + str(tagDF.loc[tag,parent])
            ret = ret + math.log(tagDF.loc[tag,parent])
        return(ret)
        
    #To get the inclusion
    def getInclusionsGivenParent(self, inputs):
        level = inputs[0]
        tagDF = inputs[1]
        parent = inputs[2]
        debug = inputs[3]
        if debug == 1:
            print 'Beginning Level...........'
            print level
        inTags = [x[0] for x in level[1:]]
        if u'' in inTags:
            inTags.remove(u'')
            
        #Do some recursion
        for i,tag in enumerate(inTags):
            if tag in self.phraseTags or tag in self.sentenceTypes:
                if debug == 1:
                    print 'beginning recursion due to:'
                    print tag
                self.getInclusionsGivenParent([level[i+1],tagDF,tag,debug])
        
        #Add count for this level
        inTags = list(set(inTags))
        for tag in inTags:
            if type(tag) != unicode:#Some sentences contain only a word, and we won't need to add anything in that case.
                break
            if debug == 1:
                print 'incrementing: ' + tag + ' when conditioned on ' + parent
            tagDF.loc[tag,parent] += 1
            
            
    #To get the inclusions in a level recursively.
    def getInclusions(self,inputs):
        level = inputs[0]
        tagDict = inputs[1]
        debug = inputs[2]
        if debug == 1:
            print 'Beginning Level...........'
            print level
        inTags = [x[0] for x in level[1:]]
        if u'' in inTags:
            inTags.remove(u'')
            
        #Do some recursion
        for i,tag in enumerate(inTags):
            if tag in self.phraseTags or tag in self.sentenceTypes:
                if debug == 1:
                    print 'beginning recursion due to:'
                    print tag
                self.getInclusions([level[i+1],tagDict,debug])
        
        #Add count for this level
        inTags = list(set(inTags))
        for tag in inTags:
            if type(tag) != unicode:#Some sentences contain only a word, and we won't need to add anything in that case.
                break
            if debug == 1:
                print 'incrementing: ' + tag
            tagDict[tag] += 1
    
    #To find all PoS tags (pystatparser's documentation is literally non-existant)
    def getTagsRecursively(self, ss, knownTags = [], debug = 0):
        ret = knownTags
        for sentence in ss:
            for phrase in sentence:
                for element in phrase:
                    if type(element) == unicode:
                        if element not in ret:
                            ret.append(element)        
                    if type(element) == list:
                        ret.extend(self.getTagsRecursively(element))
        return(ret)
        
    #Flatten an n-dimensional list into a 1D list
    def recursiveFlatten(self, myList):
        ret = []
        for element in myList:
            if type(element) == list:
                element = self.recursiveFlatten(element)
            if type(element) == str or type(element) == unicode:
                ret.append(element)
            else:
                ret.extend(list(element))
        return(ret)
    
    #From http://stackoverflow.com/questions/4576077/python-split-text-on-sentences
    def split_into_sentences(self, text):
        if type(text) == unicode:
            text = unicode(text.encode('utf-8'), errors = 'ignore')
            text = unicodedata.normalize('NFKD',text).encode('ascii','ignore')
        caps = "([A-Z])"
        prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
        suffixes = "(Inc|Ltd|Jr|Sr|Co)"
        starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
        acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
        websites = "[.](com|net|org|io|gov)"
        text = " " + text + "  "
        text = text.replace("\n"," ")
        text = re.sub(prefixes,"\\1<prd>",text)
        text = re.sub(websites,"<prd>\\1",text)
        if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
        if 'a.m.' in text: text = text.replace('a.m.','a<prd>m<prd>')
        if 'p.m.' in text: text = text.replace('p.m.','p<prd>m<prd>')
        if '...' in text: text = text.replace('...','<prd><prd><prd>')
        text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
        text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
        text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
        text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
        text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
        text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
        text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
        if "\"" in text: text = text.replace(".\"","\".")
        if "!" in text: text = text.replace("!\"","\"!")
        if "?" in text: text = text.replace("?\"","\"?")
        text = text.replace(".",".<stop>")
        text = text.replace("?","?<stop>")
        text = text.replace("!","!<stop>")
        text = text.replace("<prd>",".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]
        return sentences
    
    #Write the parameters we have to file. This will create three files.
    #Passing the parameter "default" to this function will overwrite the \
    #parameters fit by the author.
    def storeParameters(self, target):
        try: str(target)
        except:
            print "store parameters needs to be passed a string"
            return
        f = open(target,'w')
        f.write(reduce(lambda x,y: str(x) + ',' + str(y), self.classPriors) + '\n')
        f.write(reduce(lambda x,y: str(x) + ',' + str(y), self.importantRootProbabilities.keys()) + '\n')
        f.write(reduce(lambda x,y: str(x) + ',' + str(y), self.importantRootProbabilities.values()) + '\n')
        f.write(reduce(lambda x,y: str(x) + ',' + str(y), self.regularRootProbabilities.keys()) + '\n')
        f.write(reduce(lambda x,y: str(x) + ',' + str(y), self.regularRootProbabilities.values()) + '\n')
        f.write(reduce(lambda x,y: str(x) + ',' + str(y), self.importantMultiplictyParameter.keys()) + '\n')
        f.write(reduce(lambda x,y: str(x) + ',' + str(y), self.importantMultiplictyParameter.values()) + '\n')
        f.write(reduce(lambda x,y: str(x) + ',' + str(y), self.regularMultiplictyParameter.keys()) + '\n')
        f.write(reduce(lambda x,y: str(x) + ',' + str(y), self.regularMultiplictyParameter.values()) + '\n')
        #f.write(reduce(lambda x,y: str(x) + ',' + str(y), self.alpha) + '\n')Good for bayse
        #f.write(reduce(lambda x,y: str(x) + ',' + str(y), self.beta) + '\n')
        f.write(str(self.alpha) + '\n')
        f.write(str(self.beta) + '\n')
        f.write(reduce(lambda x,y: str(x) + ',' + str(y), self.tags) + '\n')
        f.write(reduce(lambda x,y: str(x) + ',' + str(y), self.sentenceTypes) + '\n')
        f.write(reduce(lambda x,y: str(x) + ',' + str(y), self.phraseTags) + '\n')
        f.close()
        self.importantCondPresenceProbs.to_csv(target + "I.csv")
        self.regularCondPresenceProbs.to_csv(target + "R.csv")
    
    #Load parameters from file. Simply provide it with the name you provided \
    #to storeParameters. The argument "default" will load the parameters \
    #fit by the author.
    def loadParameters(self, target):
        try: str(target)
        except:
            print "load parameters needs to be passed a string"
            return
        f = open(target,'r')
        groups = [x.split(',') for x in f.read().split('\n')]
        self.classPriors = [float(x) for x in groups[0]]
        self.importantRootProbabilities = dict(zip(groups[1],[float(x) for x in groups[2]]))
        self.regularRootProbabilities = dict(zip(groups[3],[float(x) for x in groups[4]]))
        self.importantMultiplictyParameter = dict(zip(groups[5],[float(x) for x in groups[6]]))
        self.regularMultiplictyParameter = dict(zip(groups[7],[float(x) for x in groups[8]]))
        #self.alpha = groups[10]#comin with the bayes update
        #self.beta = groups[11]
        self.alpha = float(groups[9][0])
        self.beta = float(groups[10][0])
        self.tags = groups[11]
        self.sentenceTypes = groups[12]
        self.phraseTags = groups[13]
        f.close()
        self.importantCondPresenceProbs = pd.read_csv(target + 'I.csv', index_col = 0)
        self.regularCondPresenceProbs = pd.read_csv(target + 'R.csv', index_col = 0)