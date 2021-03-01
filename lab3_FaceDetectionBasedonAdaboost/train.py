import matplotlib.pyplot as plt #plt 用于显示图片
import matplotlib.image as mpimg #mpimg 用于读取图片
import numpy as np
#from scipy import misc  #对图像进行缩放
from feature import NPDFeature
from sklearn.tree import DecisionTreeClassifier
import pickle
from skimage.transform import resize

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

sampleSize = 1000
maxIteration = 20

img =[]
img_label = []
img_features = []
img_label_train, img_label_validation = [],[]
img_features_train, img_features_validation = [],[]


# Load data set data and convert the images into grayscale images with size of 24 * 24 and label them
def load_img_data():
    # img_face
    for i in range(0,int(sampleSize/2)):
        image = mpimg.imread("./datasets/original/face/face_"+"{:0>3d}".format(i)+".jpg")#3d表示三个宽度的十进制表示 用0补齐
        image_gray = rgb2gray(image)
        image_gray_scaled = resize(image_gray, output_shape=(24, 24))
        img.append(image_gray_scaled) 
        img_label.append(1)
    #img_nonface
        image = mpimg.imread("./datasets/original/nonface/nonface_"+"{:0>3d}".format(i)+".jpg")
        image_gray = rgb2gray(image)
        image_gray_scaled = resize(image_gray,output_shape=(24, 24))
        img.append(image_gray_scaled)
        img_label.append(-1)
   

def showImg(item):
    plt.imshow(item)

#convert the images into grayscale images 0.299R+0.587G+0.114B   
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])#高维切片：...省略所有的冒号，等同于[:,:,:3]
#Processing data set data to extract NPD features.
def extra_img_features():
    for i in range(0,len(img)):
        f = NPDFeature(img[i])
        features = f.extract()
        img_features.append(features)
        
def get_accuracy(pred, y):
    return sum(pred == y) / float(len(y))

def get_error_rate(pred,y):
    return sum(pred != y) / float(len(y))

if __name__ == "__main__":

    load_img_data()    

    with open('data', "wb") as f:#write binary file
       extra_img_features()
       pickle.dump(img_features, f)

    #the data set is divided into training set and validation set            
    img_label_train = img_label[0:int(sampleSize*0.7)]
    img_label_validation = img_label[int(sampleSize*0.7):]
    img_features_train = img_features[0:int(sampleSize*0.7)]
    img_features_validation = img_features[int(sampleSize*0.7):]

    #initialize training set weights , each training sample is given the same weight.  
    weights = np.ones(len(img_features_train)) / len(img_features_train)
    #Training a base classifier
    clf_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)

    hypothesis_train, hypothesis_validation = [], []
    alpha_m = []

    prediction_train = np.zeros(len(img_features_train),dtype=np.int32)
    prediction_validation = np.zeros(len(img_features_validation),dtype=np.int32)
    
    accuracy_train,accuracy_validation = [] ,[]
    
    for i in range(0, maxIteration):
        print("Number of decision trees:",i+1)
        print(len(img_features_train), len(img_label_train))
        clf_tree.fit(img_features_train, img_label_train, sample_weight=weights)#fit base learner
        hypothesis_train.append (clf_tree.predict(img_features_train) )
        hypothesis_validation.append(clf_tree.predict(img_features_validation))
            
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


        
    plt.xlabel("Number of Decision Trees")
    plt.ylabel("Accuracy")
    plt.plot(accuracy_train, label ="train")
    plt.plot(accuracy_validation, label="validation")
    plt.legend(loc="lower right")
    plt.savefig('plot1.png', format='png')
#use classification_report () of the sklearn.metrics library function writes predicted result to classifier_report.txt
f = open("./report.txt", 'w+')  
print(classification_report(img_label_validation,np.sign(prediction_validation)),file=f)
f.close()
