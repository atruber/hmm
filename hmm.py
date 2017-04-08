"""Hidden Markov Model sequence tagger

"""
from classifier import Classifier
import numpy, copy, math

class HMM(Classifier):
        
    def get_model(self): return None
    def set_model(self, model): pass

    model = property(get_model, set_model)
    
    def __init__(self):
        self.feature_count_table = [] #dummy initialization
        self.transition_count_table = []
        self.emission_matrix = []
        self.transition_matrix = []
        self.feature_idx = {}
        self.idx_feature = {}
        self.label_idx = {}
        self.idx_label = {}
        self.starts = []
        self.stops = []
        
    def make_indices(self, instance_list, update=False):
        """set up mapping of labels and features to integers to be used as indices"""
        features = set()
        features.add('UNK')
        labels = set()
        for instance in instance_list:
            for i in range(len(instance.data)):
                features.add(str(instance.data[i]))
                labels.add(instance.label[i])
        features = list(features)
        labels = list(labels)
            
        for i in range(len(features)):
            self.feature_idx[features[i]] = i
            self.idx_feature[i] = features[i]
        
        for i in range(len(labels)):
            self.label_idx[labels[i]] = i
            self.idx_label[i] = labels[i] 
       
        
    def _collect_counts(self, instance_list):
        """Collect counts necessary for fitting parameters. This function should update self.transtion_count_table
        and self.feature_count_table based on this new given instance
        Returns None"""
        
        #loop through and get counts of both transitions and label/feature pairs
        for instance in instance_list:
            #fencepost last feature pair and at start/stop labels
            a = instance.label[0]
            j = instance.label[-1]
            k = str(instance.data[-1])
            self.feature_count_table[self.label_idx[j]][self.feature_idx[k]] += 1     
            self.starts[self.label_idx[a]] += 1
            self.stops[self.label_idx[j]] += 1
            
            i = 0
            while i < len(instance.label)-1:
                x = instance.label[i]
                y = str(instance.data[i])
                z = instance.label[i+1]
                self.feature_count_table[self.label_idx[x]][self.feature_idx[y]] += 1  
                self.transition_count_table[self.label_idx[x]][self.label_idx[z]] +=1
                i += 1
                
    def unk_probs(self):
        #use infrequent words to estimate UNK probs
        for i in range(len(self.label_idx)):
            self.emission_matrix[i,self.feature_idx['UNK']] = math.log((self.feature_count_table[i,:] < 3).sum() / self.feature_count_table[i,:].sum())
         
    def train(self, instance_list):
        """Fit parameters for hidden markov model. Update codebooks from the given data to be consistent with
        the probability tables. Transition matrix and emission probability matrix
        will then be populated with the maximum likelihood estimate of the appropriate parameters
        Method calls count method, then divides by total counts to obtain emission and transition probs
        Returns None"""
        self.make_indices(instance_list)  
        labs = len(self.label_idx.items())
        feats = len(self.feature_idx.items())  
        
        #initialize matrices to proper size
        self.transition_count_table = numpy.ones((labs,labs))
        self.feature_count_table = numpy.ones((labs,feats))
        self.starts = numpy.ones((labs,1))
        self.stops = numpy.ones((labs,1))
        
        self._collect_counts(instance_list)
        
        #calculate transition and emission probabilities
        #prob = #instances / total instances in class
        self.transition_matrix = copy.deepcopy(self.transition_count_table)
        self.emission_matrix = copy.deepcopy(self.feature_count_table)
        for x in range (len(self.label_idx)):
            self.transition_matrix[x,:] = self.transition_matrix[x,:] / self.transition_matrix[x,:].sum()
            self.emission_matrix[x,:] = self.emission_matrix[x,:] / self.emission_matrix[x,:].sum()
            
        self.transition_matrix = numpy.log(self.transition_matrix)
        self.emission_matrix = numpy.log(self.emission_matrix)
        self.unk_probs()
      
        #calculate start and termination probabilities
        sta = self.starts.sum()
        sto = self.stops.sum()
        self.starts = numpy.log(self.starts/sta)
        self.stops = numpy.log(self.stops/sto)
        
        
    def classify(self, instance):
        """Viterbi decoding algorithm. Wrapper for running the Viterbi algorithm
        We can then obtain the best sequence of labels from the backtrace pointers matrix
        Returns a list of labels e.g. ['B','I','O','O','B']"""
        backtrace_pointers = self.dynamic_programming_on_trellis(instance, False)
        bp = backtrace_pointers[1]
        trellis = backtrace_pointers[0]
        best_sequence = []
        x = numpy.argmax(trellis[:,-1])
        y = bp.shape[1]-1
        while y > 0:
            best_sequence.insert(0,self.idx_label[x])
            curr = bp[x,y]
            if type(curr) is tuple:
                x = curr[0]
                y = curr[1]
            else:
                pass
        best_sequence.insert(0,self.idx_label[x])
        
        return best_sequence

    def compute_observation_loglikelihood(self, instance):
        """Compute and return log P(X|parameters) = loglikelihood of observations"""
        trellis = self.dynamic_programming_on_trellis(instance, True)
        loglikelihood = max(trellis[:,-1])
        return loglikelihood

    def dynamic_programming_on_trellis(self, instance, run_forward_alg=True):
        """Run Forward algorithm or Viterbi algorithm. This function uses the trellis to implement dynamic
        programming algorithm for obtaining the best sequence of labels given the observations
        Returns trellis filled up with the forward probabilities and backtrace pointers for finding the best sequence """
        
        numlab = len(self.label_idx)
        numstates = len(instance.data)
        #Initialize trellis and backtrace pointers 
        trellis = numpy.zeros((numlab,numstates))
        backtrace_pointers = numpy.zeros((numlab,numstates), dtype=tuple)
        
        #Traverse through the trellis here
        #fill in probs for q1 (emission prob*initial prob)
        for x in xrange(0, numlab):
            trellis[x,0] = self.emission_matrix[x][0] + self.starts[x]
        
        #fill in rest of trellis
        #iterate through each observation
        for j in xrange(1,numstates):
            feat = str(instance.data[j])
            if feat in self.feature_idx:
                pass
            else:
                feat = 'UNK'
            feat_idx = self.feature_idx[feat]    
            #print instance.data[j]
            #iterate through each state/label
            for i in xrange(0,numlab):
                maxprob = -100000 
                lab_idx = i
                #iterate through each potential previous label:
                for z in xrange(0,numlab):
                    prev_lab = self.label_idx[self.idx_label[z]]
                    ep = self.emission_matrix[lab_idx][feat_idx]
                    tp = self.transition_matrix[prev_lab][lab_idx]
                    pp = trellis[z][j-1] 
                    currprob = ep + tp + pp 
                    if currprob > maxprob:
                        maxprob = currprob
                        backtrace_pointers[i][j] = (eval('prev_lab'),eval('j-1'))
                    trellis[i][j] = maxprob
        
        #apply termination probs
        for x in xrange(0, numlab):
            trellis[x,numstates-1] = self.emission_matrix[x][numstates-1] + self.stops[x]
                                          

        if run_forward_alg == False:
            return (trellis, backtrace_pointers)
        return trellis

#    def train_semisupervised(self, unlabeled_instance_list, labeled_instance_list=None):
#        """Baum-Welch algorithm for fitting HMM from unlabeled data (EXTRA CREDIT)
#
#        The algorithm first initializes the model with the labeled data if given.
#        The model is initialized randomly otherwise. Then it runs 
#        Baum-Welch algorithm to enhance the model with more data.
#
#        Add your docstring here explaining how you implement this function
#
#        Returns None
#        """
#        if labeled_instance_list is not None:
#            self.train(labeled_instance_list)
#        else:
#            #TODO: initialize the model randomly
#            pass
#        while True:
#            #E-Step
#            self.expected_transition_counts = numpy.zeros((1,1))
#            self.expected_feature_counts = numpy.zeros((1,1))
#            for instance in instance_list:
#                (alpha_table, beta_table) = self._run_forward_backward(instance)
#                #TODO: update the expected count tables based on alphas and betas
#                #also combine the expected count with the observed counts from the labeled data
#            #M-Step
#            #TODO: reestimate the parameters
#            if self._has_converged(old_likelihood, likelihood):
#                break
#
#    def _has_converged(self, old_likelihood, likelihood):
#        """Determine whether the parameters have converged or not (EXTRA CREDIT)
#
#        Returns True if the parameters have converged.    
#        """
#        return True
#
#    def _run_forward_backward(self, instance):
#        """Forward-backward algorithm for HMM using trellis (EXTRA CREDIT)
#    
#        Fill up the alpha and beta trellises (the same notation as 
#        presented in the lecture and Martin and Jurafsky)
#        You can reuse your forward algorithm here
#
#        return a tuple of tables consisting of alpha and beta tables
#        """
#        alpha_table = numpy.zeros((1,1))
#        beta_table = numpy.zeros((1,1))
#        #TODO: implement forward backward algorithm right here
#
#        return (alpha_table, beta_table)
#
