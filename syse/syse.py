import pandas as pd
from stat_parser import Parser
import re
import numpy as np
import unicodedata
import math
from pkg_resources import resource_filename

###
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
        send = resource_filename(__name__, 'default.dat')
        print(send)
        self.loadParameters(send)
        
    ####Supervised Training.
    #trainingSentences: sentences on which to train (Must already be parsed)
    #labels: corresponding binary (1,0) labels.
    #alpha: laplace/additive smoothing parameter (default = 1)
    def train(self, trainingSentences, labels, binomHyperParams = [0.5,0.5], poissonHyperParams = [0.0001,0.005], debug = 0):
        if debug > -1:
            print
            print '**********************************************************'
            print '                   SySE V 1.1 '
            print 'Beginning Training Sequence with ' + \
                str(len(trainingSentences)) + ' training sentences...'
            print '**********************************************************'
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
                trainingSentences = [self.parser.parse(x) for x in trainingSentences]
            except:
                print 'This environment should have pystatparser installed ' + \
                'in order to train on unparsed sentences.'
                print 'Parameters could not be fit'
                print 'Exiting...'
                return
            
        
        ####Initialization
        #Save hyperparameters
        self.binomHyperParams = binomHyperParams
        self.poissonHyperParams = poissonHyperParams
        
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
        self.importantRootProbabilities = dict(zip(list(self.sentenceTypes),[binomialParamDist(self.binomHyperParams) for x in range(0,len(list(self.sentenceTypes)))]))
        self.regularRootProbabilities = dict(zip(list(self.sentenceTypes),[binomialParamDist(self.binomHyperParams) for x in range(0,len(list(self.sentenceTypes)))]))
        
        #Get the count of each sentence type in I
        for sentence in importantSentences:
            #Make sure we get what we expect
            if type(sentence[0]) != unicode:
                print "We are looking for a non-unicode sentence type. exiting..."
                break
                return            
            #if it isn't in the list yet, add it.
            self.importantRootProbabilities[sentence[0]].update(1)
            for sentence1 in importantSentences:
                if sentence1 != sentence:
                    self.importantRootProbabilities[sentence1[0]].update(False)
        
        #Get the count of each sentence type in R
        for sentence in regularSentences:
            #Make sure we get what we expect
            if type(sentence[0]) != unicode:
                print "We are looking for a non-unicode sentence type. exiting..."
                break
                return            
            #if it isn't in the list yet, add it.
            self.regularRootProbabilities[sentence[0]].update(1)
            for sentence1 in importantSentences:
                if sentence1 != sentence:
                    self.regularRootProbabilities[sentence1[0]].update(False)
        
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
        #To store poisson beliefs
        self.importantMultiplictyParameter = dict(zip(list(self.tags), [poissonParamDist(self.poissonHyperParams) for x in range(0,len(list(self.tags)))]))#For storing parameter estimates.
        self.regularMultiplictyParameter = dict(zip(list(self.tags), [poissonParamDist(self.poissonHyperParams) for x in range(0,len(list(self.tags)))]))#For storing parameter estimates.
        
        #Get Inclusion for I
        for sentence in importantSentences:
            self.getInclusions([sentence,self.importantMultiplictyParameter,debug>=2])
        
        #Get Inclusion for R
        for sentence in regularSentences:
            self.getInclusions([sentence,self.regularMultiplictyParameter,debug>=2])
        
        #Get Counts for I
        for sentence in importantSentences:
            flat = self.recursiveFlatten(sentence)
            currentTags = filter(lambda x: type(x)==unicode, flat)
            for tag in currentTags[1:]:
                self.importantMultiplictyParameter[tag].updateCount(1)
        
        #Get Counts for R
        for sentence in regularSentences:
            flat = self.recursiveFlatten(sentence)
            currentTags = filter(lambda x: type(x)==unicode, flat)
            for tag in currentTags[1:]:
                self.regularMultiplictyParameter[tag].updateCount(1)
        
        ####YOUNEED TO GO OVER THIS AGAIN
        #Estimate Parameters for I
        for tag in self.importantMultiplictyParameter.keys():
            if (self.importantMultiplictyParameter[tag].alpha > 1):
                self.importantMultiplictyParameter[tag].updateCount(-1)
                
        #Estimate Parameters for R
        for tag in self.regularMultiplictyParameter.keys():
            if (self.regularMultiplictyParameter[tag].alpha > 1):
                self.regularMultiplictyParameter[tag].updateCount(-1)
        
            if debug > 0:
                print '*********************************************************'
                print ' Estimation for Multiplicity Parameters '
                print '*********************************************************'
                print
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
        #For important phrases
        self.importantCondPresenceProbs = np.zeros([len(self.tags),len(self.phraseTags) + len(self.sentenceTypes)])
        self.importantCondPresenceProbs = pd.DataFrame(self.importantCondPresenceProbs).applymap(lambda x: binomialParamDist(self.binomHyperParams))
        self.importantCondPresenceProbs.columns = list(self.sentenceTypes) + list(self.phraseTags)
        self.importantCondPresenceProbs.index = list(self.tags)
        
        #For regularPhrases
        self.regularCondPresenceProbs = np.zeros([len(self.tags),len(self.phraseTags) + len(self.sentenceTypes)])
        self.regularCondPresenceProbs = pd.DataFrame(self.regularCondPresenceProbs).applymap(lambda x: binomialParamDist(self.binomHyperParams))
        self.regularCondPresenceProbs.columns = list(self.sentenceTypes) + list(self.phraseTags)
        self.regularCondPresenceProbs.index = list(self.tags)
        
        #Count Conditional Inclusions for Important Sentences
        for sentence in importantSentences:
            self.getInclusionsGivenParent([sentence,self.importantCondPresenceProbs,sentence[0],debug>=2])
        
        #Count Conditional Inclusions for Regular Sentences
        for sentence in regularSentences:
            self.getInclusionsGivenParent([sentence,self.regularCondPresenceProbs,sentence[0],debug>=2])
        
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
    def classify(self, sentence, debug = 0, varianceExponent = 0):
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
            self.importantRootProbabilities[sentence[0]] = self.biomialParamDist(self.binomHyperParams)
        if sentence[0] not in self.regularRootProbabilities:
            self.regularRootProbabilities[sentence[0]] = self.biomialParamDist(self.binomHyperParams)
        
        #Deal with new non-root tag types
        flat = self.recursiveFlatten(sentence)
        flat = filter(lambda x: type(x) == unicode, flat)
        for i,tag in enumerate(flat):
            if tag not in self.tags:
                #Set a priori beliefs for multiplicity parameters
                self.importantMultiplictyParameter[tag] = self.poissonParamDist(self.poissonHyperParams)
                self.regularMultiplictyParameter[tag] = self.poissonParamDist(self.poissonHyperParams)
                
                #Set a priori beliefs for conditional presence parameters being contained by anything else
                self.importantCondPresenceProbs.loc[tag] = [self.biomialParamDist(self.binomHyperParams) for x in self.importantCondPresenceProbs.columns]
                self.regularCondPresenceProbs.loc[tag] = [self.biomialParamDist(self.binomHyperParams) for x in self.regularCondPresenceProbs.columns]
                if type(flat[i+1])==unicode:
                    #Set a priori beliefs for conditional presence parameters containing other things
                    self.importantCondPresenceProbs[tag] = [self.biomialParamDist(self.binomHyperParams) for x in self.regularCondPresenceProbs.index]
                    self.regularCondPresenceProbs[tag] = [self.biomialParamDist(self.binomHyperParams) for x in self.regularCondPresenceProbs.index]
        
        ##Get P(x|y = Important) 
        PxGy1 = math.log(self.importantRootProbabilities[sentence[0]].getMean()) 
        PxGy1 = PxGy1 / self.importantRootProbabilities[sentence[0]].getVar()**varianceExponent
        PxGy1 += self.getConditionalLevelProbability([sentence,self.importantCondPresenceProbs,self.importantMultiplictyParameter,sentence[0],varianceExponent,debug>=2])
        
        ##Get P(x|y = REGULAR) 
        PxGy0 = math.log(self.regularRootProbabilities[sentence[0]].getMean())
        PxGy0 = PxGy0/self.regularRootProbabilities[sentence[0]].getVar()**varianceExponent
        PxGy0 += self.getConditionalLevelProbability([sentence,self.regularCondPresenceProbs,self.regularMultiplictyParameter,sentence[0],varianceExponent,debug>=2])
        
        #Get priors in a log form:
        Py1 = math.log(self.classPriors[1])
        Py0 = math.log(self.classPriors[0])
        
        #Get log Probabilities of each class through Bayes' Rule
        Py1Gx = PxGy1+Py1
        Py0Gx = PxGy0+Py0
        
        print Py1Gx
        print Py0Gx
        
        #Derive softmax shift parameter for very small probabilities.
        shift = 0
        if min([Py1Gx,Py0Gx]) < -20:
            shift = -1*min([Py1Gx,Py0Gx]) - 20
            print 'Very low conditional probabity. Using a shift of ' + str(shift)
            print 'Originial Probabilities were ' + str([Py1Gx,Py0Gx])
            print 'New probabilities are ' + str([x + shift for x in [Py1Gx,Py0Gx]])
        
        #SoftMax probabilities
        denom = math.log(math.e**(shift + Py1Gx) + math.e**(shift + Py0Gx))
        print denom
        
        sPy1Gx = shift + Py1Gx-denom
        sPy0Gx = shift + Py0Gx-denom
        
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
        varExp = inputs[4]
        debug = inputs[5]
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
                ret = ret + self.getConditionalLevelProbability([level[i+1],tagDF,mult,tag,varExp,debug])
        
        #Do multiplicity for this level
        for tag in inTags:
            x = inTags.count(tag)
            mu = mult[tag].getMean()
            ret = ret + math.log((math.exp(-mu) * mu**x / math.factorial(x)))/mult[tag].getVar()**varExp
        
        #Do presence for this level
        inTags = list(set(inTags))
        for tag in inTags:
            if type(tag) != unicode:#Some sentences contain only a word, and we won't need to add anything in that case.
                print 'Breaking Due to non-unicode tag in getConditionalLevelProbability!'
                print tag
                break
            if debug == 1:
                print 'Probability of ' + tag + ' given ' + parent + ' is ' + str(tagDF.loc[tag,parent])
            ret = ret + math.log(tagDF.loc[tag,parent].getMean())/tagDF.loc[tag,parent].getVar()**varExp
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
        
        #Update tags on this level
        inTags = list(set(inTags))
        for tag in inTags:
            if type(tag) != unicode:#Some sentences contain only a word, and we won't need to add anything in that case.
                break
            if debug == 1:
                print 'incrementing: ' + tag + ' when conditioned on ' + parent
            tagDF.loc[tag,parent].update(True)
        #Update tags not on this level
        for tag in self.tags:
            if tag not in inTags:
                tagDF.loc[tag,parent].update(False)
            
            
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
            tagDict[tag].incrementTrials()
    
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
        if "”" in text: text = text.replace(".”","”.")
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
        f.write(reduce(lambda x,y: str(x) + '/-_-/' + str(y), self.classPriors) + '\n')
        f.write(reduce(lambda x,y: str(x) + '/-_-/' + str(y), self.importantRootProbabilities.keys()) + '\n')
        f.write(reduce(lambda x,y: str(x) + '/-_-/' + str(y), [x.store() for x in self.importantRootProbabilities.values()]) + '\n')
        f.write(reduce(lambda x,y: str(x) + '/-_-/' + str(y), self.regularRootProbabilities.keys()) + '\n')
        f.write(reduce(lambda x,y: str(x) + '/-_-/' + str(y), [x.store() for x in self.regularRootProbabilities.values()]) + '\n')
        f.write(reduce(lambda x,y: str(x) + '/-_-/' + str(y), self.importantMultiplictyParameter.keys()) + '\n')
        f.write(reduce(lambda x,y: str(x) + '/-_-/' + str(y), [x.store() for x in self.importantMultiplictyParameter.values()]) + '\n')
        f.write(reduce(lambda x,y: str(x) + '/-_-/' + str(y), self.regularMultiplictyParameter.keys()) + '\n')
        f.write(reduce(lambda x,y: str(x) + '/-_-/' + str(y), [x.store() for x in self.regularMultiplictyParameter.values()]) + '\n')
        #f.write(reduce(lambda x,y: str(x) + ',' + str(y), self.alpha) + '\n')Good for bayse
        #f.write(reduce(lambda x,y: str(x) + ',' + str(y), self.beta) + '\n')
        f.write(str(self.binomHyperParams[0]) + '/-_-/' +  str(self.binomHyperParams[1]) + '\n')
        f.write(str(self.poissonHyperParams[0]) + '/-_-/' + str(self.poissonHyperParams[1]) + '\n')
        f.write(reduce(lambda x,y: str(x) + '/-_-/' + str(y), self.tags) + '\n')
        f.write(reduce(lambda x,y: str(x) + '/-_-/' + str(y), self.sentenceTypes) + '\n')
        f.write(reduce(lambda x,y: str(x) + '/-_-/' + str(y), self.phraseTags) + '\n')
        #Element-wise store parameters
        ICP = []
        for i in self.importantCondPresenceProbs.index:
            for j in self.importantCondPresenceProbs.columns:
                ICP.append(self.importantCondPresenceProbs.loc[i,j].store())
        RCP = []
        for i in self.regularCondPresenceProbs.index:
            for j in self.regularCondPresenceProbs.columns:
                RCP.append(self.regularCondPresenceProbs.loc[i,j].store())
        f.write(reduce(lambda x,y: str(x) + '/-_-/' + str(y), ICP) + '\n')
        f.write(reduce(lambda x,y: str(x) + '/-_-/' + str(y), RCP) + '\n')
        f.close()
        
    #Load parameters from file. Simply provide it with the name you provided \
    #to storeParameters. The argument "default" will load the parameters fit \
    #by the author.
    def loadParameters(self, target):
        try: str(target)
        except:
            print "load parameters needs to be passed a string"
            return
        f = open(target,'r')
        groups = [x.split('/-_-/') for x in f.read().split('\n')]
        self.classPriors = [float(x) for x in groups[0]]
        self.importantRootProbabilities = dict(zip([unicode(x) for x in groups[1]],[binomialParamDist().load(x) for x in groups[2]]))
        self.regularRootProbabilities = dict(zip([unicode(x) for x in groups[3]],[binomialParamDist().load(x) for x in groups[4]]))
        self.importantMultiplictyParameter = dict(zip([unicode(x) for x in groups[5]],[poissonParamDist().load(x) for x in groups[6]]))
        self.regularMultiplictyParameter = dict(zip([unicode(x) for x in groups[7]],[binomialParamDist().load(x) for x in groups[8]]))
        #self.alpha = groups[10]#comin with the bayes update
        #self.beta = groups[11]
        self.binomHyperParams = [float(x) for x in groups[9]]
        self.poissonHyperParams = [float(x) for x in groups[10]]
        self.tags = [unicode(x) for x in groups[11]]
        self.sentenceTypes = [unicode(x) for x in groups[12]]
        self.phraseTags = [unicode(x) for x in groups[13]]
        
        #Unpack dataframes
        self.importantCondPresenceProbs = np.zeros([len(self.tags),len(self.phraseTags) + len(self.sentenceTypes)])
        self.importantCondPresenceProbs = pd.DataFrame(self.importantCondPresenceProbs)
        self.importantCondPresenceProbs.columns = list(self.sentenceTypes) + list(self.phraseTags)
        self.importantCondPresenceProbs.index = list(self.tags)
        
        self.regularCondPresenceProbs = np.zeros([len(self.tags),len(self.phraseTags) + len(self.sentenceTypes)])
        self.regularCondPresenceProbs = pd.DataFrame(self.regularCondPresenceProbs)
        self.regularCondPresenceProbs.columns = list(self.sentenceTypes) + list(self.phraseTags)
        self.regularCondPresenceProbs.index = list(self.tags)
        
        for i,row in enumerate(self.importantCondPresenceProbs.index):
            for j,column in enumerate(self.importantCondPresenceProbs.columns):
                self.importantCondPresenceProbs.loc[row,column] = binomialParamDist().load(groups[14][i*len(self.importantCondPresenceProbs.columns) + j])
        
        for i,row in enumerate(self.regularCondPresenceProbs.index):
            for j,column in enumerate(self.regularCondPresenceProbs.columns):
                self.regularCondPresenceProbs.loc[row,column] = binomialParamDist().load(groups[15][i*len(self.regularCondPresenceProbs.columns) + j])
        
        f.close()
    
    def binomialParamDist(params):
        return(binomialParamDist(params))
    
    def poissonParamDist(params):
        return(poissonParamDist(params))

class binomialParamDist:
    def __init__(self, prior = [0.5,0.5]):
        self.prior = [float(x) for x in prior]
        self.alpha = self.prior[0]
        self.beta = self.prior[1]
    
    def update(self, update):
        try:
            if update:
                self.alpha += 1
            else:
                self.beta += 1
        except:
            print 'Binomial Parameter was asked to sequentially update on a' +\
                ' non-boolean datum.'
            print 'That\'s kind of a serious problem.'
        
    #Get the mean of a beta
    def getMean(self):
        return(self.alpha/(self.beta + self.alpha))
    
    #Get the var of a beta
    def getVar(self):
        num = self.alpha*self.beta
        denom = ((self.alpha+self.beta)**2)*(self.alpha+self.beta+1)
        return(num/denom)
    
    def __str__(self):
        return('Beta Dist with mean ' + str(self.getMean()) + ' and variance ' + str(self.getVar()) )
    
    def store(self):
        return(str(self.alpha) + '-' + str(self.beta))
    
    def load(self, target):
        self.alpha = float(target.split('-')[0])
        self.beta = float(target.split('-')[1])
        return(self)
        

class poissonParamDist:#gamma distribution, both prior and posterior
    def __init__(self, prior = [0.5,0.5]):
        self.prior = [float(x) for x in prior]
        self.alpha = self.prior[0]
        self.beta = self.prior[1]
    
    def updateCount(self, count):
        try:
            self.alpha += count
        except:
            print 'Poisson Parameter was asked to sequentially update on a' +\
                ' non-numeric datum.'
            print 'That\'s kind of a serious problem.'
    
    def incrementTrials(self):
        self.beta += 1
        
    #Get the mean of a gamma
    def getMean(self):
        return(self.alpha/self.beta)
    
    #Get the var of a gamma
    def getVar(self):
        return(self.alpha/(self.beta**2))
        
    def pdf(self, k):
        return()
    
    def __str__(self):
        return('Gamma Dist with mean ' + str(self.getMean()) + ' and variance ' + str(self.getVar()) )
    
    def store(self):
        return(str(self.alpha) + '-' + str(self.beta))
    
    def load(self, target):
        self.alpha = float(target.split('-')[0])
        self.beta = float(target.split('-')[1])
        return(self)