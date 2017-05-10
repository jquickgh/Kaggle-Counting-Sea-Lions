import cv2
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

def show_banner():
    plt.figure(figsize=(11, 5))
    plt.imshow(mpimg.imread('./github_pics/sea_lion_banner2.jpg'))
    plt.axis('off')
    plt.show()
    plt.figure(figsize=(11, 5))
    plt.imshow(mpimg.imread('./github_pics/sea_lion_scoring3.png'))
    plt.axis('off')
    plt.show() 

def show_yolo():
    #plt.figure(figsize=(11, 5))
    #plt.imshow(mpimg.imread('./github_pics/yolo_model.png'))
    #plt.axis('off')
    #plt.show()

    plt.figure(figsize=(16, 10))
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(mpimg.imread('./github_pics/darknet.png'))
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.imshow(mpimg.imread('./github_pics/yolo_map_vs_fps.png'))
    plt.show()

    plt.figure(figsize=(11, 5))
    plt.imshow(mpimg.imread('./github_pics/yolo_model2.png'))
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.imshow(mpimg.imread('./github_pics/yolo_youtube.png'))
    plt.axis('off')
    plt.show()
        
def get_correl(df):
    print ('\nCorrelation between each Category of Sea Lion:\n')
    a = df.iloc[:,0]
    b = df.iloc[:,1]
    c = df.iloc[:,2]
    d = df.iloc[:,3]
    e = df.iloc[:,4]
    print (np.corrcoef([a,b,c,d,e]),'\n')
    return (a,b,c,d,e)
    
def plot_visuals(classes):   
    a,b,c,d,e = classes
    
    plt.figure(figsize=(18,4))
    plt.subplot(1,3,1)
    plt.xlim(0,60)
    plt.ylim(0,500)
    plt.title('Males vs Females',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.scatter(a,c, color='b')
    plt.subplot(1,3,2)
    plt.xlim(0,60)
    plt.ylim(0,400)
    plt.title('Males vs Pups',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.scatter(a,e, color='g')
    plt.subplot(1,3,3)
    plt.xlim(0,500)
    plt.ylim(0,400)
    plt.title('Females vs Pups',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.scatter(c,e)
    plt.show()
    
    plt.figure(figsize=(18,4))
    plt.title('Semilog Sorted Values for each class',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=15)

    plt.semilogy(sorted(a),'r')
    plt.semilogy(sorted(b),'m')
    plt.semilogy(sorted(c),'k')
    plt.semilogy(sorted(d),'b')
    plt.semilogy(sorted(e),'g')
    plt.show()
    return None

def check_colors(df,df_color,train_img):
    color_true, color_found, color_diff, color_acc = [], [], [], []
    for color in ['red','magenta','brown','blue','green']: #,'error']:
        color_true.append(np.sum(df.loc[train_img[0]:train_img[1]-1][color]))
        color_found.append(df_color['color'].value_counts()[color])
        color_diff.append(color_true[-1] - color_found[-1])
        color_acc.append(np.round(100-np.abs((color_diff[-1]/color_true[-1])*100)))
    print ('\ncolors: red, magenta, brown, blue, green\n')
    print ('color_true: ',color_true)
    print ('color_found:',color_found)
    print ('color_diff: ',color_diff,'\n')
    print ('color accuracy:', color_acc)
    return None

def plot_rgb_colors(df_color):
    red = df_color[df_color['color'] == 'red']
    magenta = df_color[df_color['color'] == 'magenta']
    brown = df_color[df_color['color'] == 'brown']
    blue = df_color[df_color['color'] == 'blue']
    green = df_color[df_color['color'] == 'green']
    error = df_color[df_color['color'] == 'error']

    k = 0
    plt.figure(figsize=(18,14))
    for sea_lion in [(red,'r'), (magenta,'m'), (brown,'k'), (blue,'b'), (green,'g'), (error,'y')]:
    
        for i, channel in enumerate(['R','G','B']): 
            k += 1
            plt.subplot(6,3,k)
            plt.hist(sea_lion[0][:][channel], color=sea_lion[1])
            plt.xlim(0,255)
            plt.ylim(0,50)
    plt.show()
    return None

def plot_sample_images(df_blob):
    fname, shift = 47, 48
    img_1 = mpimg.imread('./input/TrainDotted/' + str(fname) + '.jpg')

    k = 0
    plt.figure(figsize=(8,16))
    for sea_lion in ['red','magenta','brown','blue','green','error']:
        for j in range(3):
            k += 1
            plt.subplot(6,3,k)
            y, x = df_blob.loc[fname][sea_lion][j]
            img_2 = img_1[y-shift:y+shift,x-shift:x+shift]
            if img_2.shape != (96,96,3):
                y, x = df_blob.loc[fname][sea_lion][j+3]
                img_2 = img_1[y-shift:y+shift,x-shift:x+shift]
            plt.imshow(img_2)
    plt.show()
    return None

def show_pic_resize():
    fnames = glob.glob('./sample_pics/*.jpg')
    pics = []
    num_pics = len(fnames)

    plt.figure(figsize=(18,6)) 
    for i in range(num_pics):
        pics.append(mpimg.imread(fnames[i]))
        plt.subplot(1,num_pics,i+1)
        plt.imshow(pics[i])
    plt.show()
    print (pics[0].shape)

    for i in range(3):
        x, y = int(pics[0].shape[1]*0.5), int(pics[0].shape[0]*0.5)
        plt.figure(figsize=(18,6)) 
        for i in range(num_pics):
            pics[i] = cv2.resize(pics[i], (x,y))
            plt.subplot(1,num_pics,i+1)
            plt.imshow(pics[i])
        plt.show()
        print (pics[0].shape)

def drive_learning_curve(df_color):
    X = df_color[['R','G','B']].values
    y = df_color[['color']].values
   
    clf = GaussianNB()
    plot_learning_curve(clf, 'GaussianNB', X, y, ylim=(0.90, 1.01))
    #clf = RandomForestClassifier()
    #plot_learning_curve(clf, 'RandomForestClassifier', X, y, ylim=(0.90, 1.01))
    #clf = LinearSVC(C=1.0)
    #plot_learning_curve(clf,'LinearSVC', X, y, ylim=(0.70, 1.01))
    return X,y
   
def plot_learning_curve(estimator, title, X, y, ylim=None): 
    print ('Plotting Learning Curve...')
    t=time.time()
    cv = StratifiedShuffleSplit(y=y, n_iter=3, test_size=0.2, random_state=0)
    train_sizes=np.linspace(.1, 1.0, 5)
    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y,  train_sizes=train_sizes, cv=cv, n_jobs=4)   
    t2 = time.time()
    print (round(t2-t, 2), 'Seconds to Plot Learning Curve...')
    print ('Done!')
        
    plt.figure(figsize=(10,2))
    plt.title('Learning Curve ('+title+')')
    if ylim is not None: plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_scores_mean, train_scores_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    test_scores_mean, test_scores_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()    
    return None

def show_augmented_data():
    print ('\nShowing augmented data transformations:')
    fname = 15
    img_1 = mpimg.imread('./input/X_TrainDotted/'+str(fname) + '.jpg')
    img_2 = np.fliplr(img_1)
    img_3 = np.flipud(img_1)
    img_4 = np.flipud(img_2)
    img_5 = np.rot90(img_1)
    img_6 = np.fliplr(img_5)
    img_7 = np.flipud(img_5)
    img_8 = np.flipud(img_6)

    plt.figure(figsize=(12,12))
    plt.subplot(4,4,1)
    plt.imshow(img_1)
    plt.subplot(4,4,2)
    plt.imshow(img_2)
    plt.subplot(4,4,3)
    plt.imshow(img_3)
    plt.subplot(4,4,4)
    plt.imshow(img_4)
    plt.subplot(4,4,5)
    plt.imshow(img_5)
    plt.subplot(4,4,6)
    plt.imshow(img_6)
    plt.subplot(4,4,7)
    plt.imshow(img_7)
    plt.subplot(4,4,8)
    plt.imshow(img_8)
    plt.show()
    return None

def plot_loss(history):
    plt.plot(history.history['categorical_accuracy'][0:])
    plt.plot(history.history['val_categorical_accuracy'][0:])
    plt.title('Model Accuracy', fontsize=15)
    plt.legend(['training set', 'validation set'], loc='lower right', fontsize=10)
    plt.ylim(0.5,1.0)
    plt.show()  
    return None

def show_nvidia_model():
    plt.figure(figsize=(20, 8))
    plt.imshow(mpimg.imread('./github_pics/nvidia_model.png'))
    plt.axis('off')
    plt.show()    


   

































