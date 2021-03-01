import pickle
from sklearn.tree import DecisionTreeClassifier

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier
        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        weak_classifier = DecisionTreeClassifier(max_depth = 1, random_state = 1)
        n_weakers_limit=20        

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).
        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        weights = np.ones(len(img_features_train)) / len(img_features_train)
        hypothesis_train, hypothesis_validation = [], []
        alpha_m = []
    
        prediction_train = np.zeros(len(img_features_train),dtype=np.int32)
        prediction_validation = np.zeros(len(img_features_validation),dtype=np.int32)
        
        accuracy_train,accuracy_validation = [] ,[]
        
        for i in range(0, self.n_weakers_limit):
            print("Number of decision trees:",i+1)
            self.weak_classifier.fit(img_features_train, img_label_train, sample_weight=weights)#fit base learner
            hypothesis_train.append (self.weak_classifier.predict(img_features_train) )
            hypothesis_validation.append(self.weak_classifier.predict(img_features_validation))
                
            miss = [int(x) for x in ( hypothesis_train[i] != img_label_train )]
            miss2 = [x if x==1 else -1 for x in miss]
    
            err_m = np.dot(weights,miss)#calculate the classification error rate
            if(err_m > 0.5):
                break
            alpha_m.append( 0.5 * np.log( (1 - err_m) / float(err_m)) )#calculate the weight of this classifier
            weights = np.multiply(weights, np.exp([float(x) * alpha_m[i] for x in miss2]))#update the weights of each data point
            weights_sum = weights.sum()
            weights = weights / weights_sum
            #output the final hypothesis
            prediction_train = prediction_train + alpha_m[i] * hypothesis_train[i]
            prediction_validation = prediction_validation + alpha_m[i] * hypothesis_validation[i]
            
            accuracy_train.append( get_accuracy(np.sign(prediction_train),img_label_train) )
            accuracy_validation.append( get_accuracy(np.sign(prediction_validation),img_label_validation) )
            print("Train Accuracy:", accuracy_train[-1])#[-1]表示取倒数第一个元素
            print("Validation Accuracy:", accuracy_validation[-1])
            if(accuracy_train[-1] == 1):#如果样本均分类成功，终止
                break
    


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.
        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.
        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.
        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        pass

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)