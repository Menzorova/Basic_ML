import numpy as np
from sklearn.datasets import load_svmlight_file
import sklearn.datasets as ds
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_svmlight_file
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.datasets import fetch_20newsgroups_vectorized, fetch_20newsgroups
import math

# TODO : Add more data sets : Both multi-class and binary as MNIST
# TODO : Add different numeric dataset or textual dataset
# TODO : Use pathlib for all paths due to multi OS system


## General utilities
def get_samples(dataset="pendigits",
                split=0,
                random_state=0,
                binary=[],
                model='other'):
    """
    [Load the given dataset . 
    Splits it as train labeled, train unlabeled and test sets.
    Uses the input random seed for splits.]
    
    Args:
        dataset (str, optional): [Dataset to use. The possible values are : ...]. Defaults to "pendigits_4_9".
        split (int, optional): [description]. Defaults to 0.
        random_state ([int]): [Random seed used to initialize the pseudo-random number generator.] Defaults to 0.
    
    Returns:
        [type]: [description]
    """

    # Load binary pendigits with images 4 and 9
    if dataset == "pendigits":
        if model == 'co-train':
            X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest = load_pendigits(
                split, random_state, binary=binary, model=model)
            return X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest

        X, Y, Xu, Yu, Xtest, Ytest = load_pendigits(split,
                                                    random_state,
                                                    binary=binary)
        return X, Y, Xu, Yu, Xtest, Ytest

    elif dataset == "iris":
        if model == 'co-train':
            X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest = load_iris(split,
                                                               random_state,
                                                               binary=binary,
                                                               model=model)
            return X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest

        X, Y, Xu, Yu, Xtest, Ytest = load_iris(split,
                                               random_state,
                                               binary=binary)
        return X, Y, Xu, Yu, Xtest, Ytest

    elif dataset == "covertype":
        if model == 'co-train':
            X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest = load_covertype(
                split, random_state, binary=binary, model=model)
            return X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest

        X, Y, Xu, Yu, Xtest, Ytest = load_covertype(split,
                                                    random_state,
                                                    binary=binary)
        return X, Y, Xu, Yu, Xtest, Ytest

    elif dataset == "voices":
        if model == 'co-train':
            X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest = load_voices(split,
                                                                 random_state,
                                                                 binary=binary,
                                                                 model=model)
            return X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest

        X, Y, Xu, Yu, Xtest, Ytest = load_voices(split,
                                                 random_state,
                                                 binary=binary)
        return X, Y, Xu, Yu, Xtest, Ytest

    elif dataset == "20_newsgroups":
        if model == 'co-train':

            X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest = load_20news(split,
                                                                 random_state,
                                                                 binary=binary,
                                                                 model=model)
            return X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest

        X, Y, Xu, Yu, Xtest, Ytest = load_20news(split,
                                                 random_state,
                                                 binary=binary)
        return X, Y, Xu, Yu, Xtest, Ytest

    elif dataset == "AirlineTweets":
        if model == 'co-train':
            X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest = load_AirlineTweets(
                split, random_state, binary=binary, model=model)

            return X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest

        X, Y, Xu, Yu, Xtest, Ytest = load_AirlineTweets(split,
                                                        random_state,
                                                        binary=binary)
        return X, Y, Xu, Yu, Xtest, Ytest

    elif dataset == "mnist_fashion":
        if model == 'co-train':
            X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest = mnist_fashion(
                split, random_state, binary=binary, model=model)
            return X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest

        X, Y, Xu, Yu, Xtest, Ytest = mnist_fashion(split,
                                                   random_state,
                                                   binary=binary)
        return X, Y, Xu, Yu, Xtest, Ytest

    elif dataset == "diabetes":
        if model == 'co-train':
            X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest = load_diabetes(
                split, random_state, binary=binary, model=model)
            return X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest

        X, Y, Xu, Yu, Xtest, Ytest = load_diabetes(split,
                                                   random_state,
                                                   binary=binary)
        return X, Y, Xu, Yu, Xtest, Ytest

    elif dataset == "mnist":
        if model == 'co-train':
            X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest = mnist(split,
                                                           random_state,
                                                           binary=binary,
                                                           model=model)
            return X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest

        X, Y, Xu, Yu, Xtest, Ytest = mnist(split, random_state, binary=binary)
        return X, Y, Xu, Yu, Xtest, Ytest

    elif dataset == "SMS_spam_ham":
        if model == 'co-train':
            X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest = load_SMS(split,
                                                              random_state,
                                                              model=model)
            return X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest

        X, Y, Xu, Yu, Xtest, Ytest = load_SMS(split, random_state)
        return X, Y, Xu, Yu, Xtest, Ytest
    
    elif dataset == "soccer":
        if model == 'co-train':
            X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest = load_soccer(split,
                                                    random_state,
                                                     model=model)
            return X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest        
        

        X, Y, Xu, Yu, Xtest, Ytest = load_soccer(split,
                                                    random_state,
                                                    binary=binary)
        return X, Y, Xu, Yu, Xtest, Ytest 
    
    elif dataset == "soccer_filt1":
        if model == 'co-train':
            X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest = load_soccer_filt1(split,
                                                    random_state,
                                                     model=model)
            return X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest        
        

        X, Y, Xu, Yu, Xtest, Ytest = load_soccer_filt1(split,
                                                    random_state,
                                                    binary=binary)
        return X, Y, Xu, Yu, Xtest, Ytest
   

    elif dataset == "soccer_filt2":
        if model == 'co-train':
            X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest = load_soccer_filt2(split,
                                                    random_state,
                                                     model=model)
            return X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest        
        

        X, Y, Xu, Yu, Xtest, Ytest = load_soccer_filt2(split,
                                                    random_state,
                                                    binary=binary)
        return X, Y, Xu, Yu, Xtest, Ytest
    
    elif dataset == "soccer_filt3":
        if model == 'co-train':
            X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest = load_soccer_filt3(split,
                                                    random_state,
                                                     model=model)
            return X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest        
        

        X, Y, Xu, Yu, Xtest, Ytest = load_soccer_filt3(split,
                                                    random_state,
                                                    binary=binary)
        return X, Y, Xu, Yu, Xtest, Ytest
    
    elif dataset == "STL10":
        if model == 'co-train':
            X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest = load_STL10(split,
                                                    random_state,
                                                     model=model)
            return X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest  
        X, Y, Xu, Yu, Xtest, Ytest = load_STL10(split, random_state, binary=binary)
        return X, Y, Xu, Yu, Xtest, Ytest
    elif dataset == "STL10_FFT2":
        if model == 'co-train':
            X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest = load_STL10(split,
                                                    random_state,
                                                     model=model, representation="FFT2")
            return X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest 
        X, Y, Xu, Yu, Xtest, Ytest = load_STL10(split, random_state, binary=binary, representation="FFT2")
        return X, Y, Xu, Yu, Xtest, Ytest
    elif dataset == "STL10_DWT2":
        if model == 'co-train':
            X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest = load_STL10(split,
                                                    random_state,
                                                     model=model, representation="DWT2")
            return X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest 
        X, Y, Xu, Yu, Xtest, Ytest = load_STL10(split, random_state, binary=binary, representation="DWT2")
        return X, Y, Xu, Yu, Xtest, Ytest

    elif dataset == "cifar10":
        if model == 'co-train':
            X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest = load_cifar10(split,
                                                    random_state,
                                                     model=model)
            return X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest 
        X, Y, Xu, Yu, Xtest, Ytest = load_cifar10(split, random_state, binary=binary)
        return X, Y, Xu, Yu, Xtest, Ytest
    elif dataset == "cifar10_FFT2":
        if model == 'co-train':
            X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest = load_cifar10(split,
                                                    random_state,
                                                     model=model, representation="FFT2")
            return X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest 
        X, Y, Xu, Yu, Xtest, Ytest = load_cifar10(split, random_state, binary=binary, representation="FFT2")
        return X, Y, Xu, Yu, Xtest, Ytest
    elif dataset == "cifar10_DWT2":
        if model == 'co-train':
            X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest = load_cifar10(split,
                                                    random_state,
                                                     model=model, representation="DWT2")
            return X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest 
        X, Y, Xu, Yu, Xtest, Ytest = load_cifar10(split, random_state, binary=binary, representation="DWT2")
        return X, Y, Xu, Yu, Xtest, Ytest
    
    elif dataset == "mushroom":
        if model == 'co-train':
            X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest = load_mushrooms(split,
                                                    random_state,
                                                     model=model)
            return X1, Y1, X1u, X2, Y2, X2u, Xtest, Ytest        
        

        X, Y, Xu, Yu, Xtest, Ytest = load_mushrooms(split,
                                                    random_state)      
        return X, Y, Xu, Yu, Xtest, Ytest

    else:
        print("The ", dataset, " dataset is not implemented.")


def horizontal_split_data_for_co_train(x, y, random_state):
    x_model_1, x_model_2, y_model_1, y_model_2 = train_test_split(
        x,
        y,
        train_size=len(x) // 2,
        test_size=len(x) - len(x) // 2,
        random_state=random_state)

    return x_model_1, x_model_2, y_model_1, y_model_2


def split_data(x, y, split, random_state, model='other'):
    """
    [Train labeled, Train unlabeled, Test splits. ]
    
    Args:
        x ([*arrays]): [Data features. We expect the first index corresponds to the sample index, the second index 
                        corresponds to the feature index.]
        y ([*array]): [Data labels]
        SizeX ([int]): [Desired size of labeled training set]
        SizeXu ([int]): [Desired size of unlabeled training set]
        SizeXtest ([int]): [Desired size of test set]
        random_state ([int]): [Random seed used to initialize the pseudo-random number generator.]
    
    Returns:
        x_l_train, y_l_train, x_u_train, y_u_train, x_u_test, y_u_test [lists]: []
    """
    x_l, x_u_test, y_l, y_u_test = train_test_split(x,
                                                    y,
                                                    train_size=split[0] +
                                                    split[1],
                                                    test_size=split[2],
                                                    random_state=random_state,
                                                    stratify=y)

    if model == 'co-train':
        x_model_1, x_model_2, y_model_1, y_model_2 = horizontal_split_data_for_co_train(
            x_l, y_l, random_state=random_state)

        x_l_train_1, x_u_train_1, y_l_train_1, y_u_train_1 = train_test_split(
            x_model_1,
            y_model_1,
            train_size=split[0] // 2,
            #test_size=math.floor(split[1]//2),
            random_state=random_state)
        x_l_train_2, x_u_train_2, y_l_train_2, y_u_train_2 = train_test_split(
            x_model_2,
            y_model_2,
            train_size=split[0] // 2,
            #test_size=math.floor(split[1]//2),
            random_state=random_state)
        print(f'Size of labeled data {x_l_train_1.shape}')
        print(f'Size of unlabeled data {x_u_train_1.shape}')
        print(f'Size of tets data {x_u_test.shape}')
        return x_l_train_1, y_l_train_1, x_u_train_1, x_l_train_2, y_l_train_2, x_u_train_2, x_u_test, y_u_test

    x_l_train, x_u_train, y_l_train, y_u_train = train_test_split(
        x_l,
        y_l,
        train_size=split[0],
        test_size=split[1],
        random_state=random_state,
        stratify=y_l)

    print(f'Size of labeled data {x_l_train.shape}')
    print(f'Size of unlabeled data {x_u_train.shape}')
    print(f'Size of tets data {x_u_test.shape}')
    return x_l_train, y_l_train, x_u_train, y_u_train, x_u_test, y_u_test


### Pendigits
def read_pendigits(path_to_data, binary):
    """
    [Loads pendigit dataset.
    If binary == [], then it loads all the classes.
    Otherwise if binary = [4,9], it loads samples belonging to classes 4 and 9 only.]
    
    Args:
        path_to_data ([type]): [description]
        binary ([[] or list of size 2]): [    If binary == [], then it loads all the classes.
        Otherwise it loads the samples belonging to the classes whose labels are in the list.]]
    
    Returns:
        x, y[*array]: [Features and labels]
    """
    df = ds.load_svmlight_file(path_to_data)
    x = df[0].todense().view(type=np.ndarray)
    y = df[1].astype(np.int)

    if (binary != []):
        # classification task is to distinguish between 4 and 9
        condition = np.isin(y, binary)
        x = x[condition, :]
        y = y[condition]
        # label is 0, when the image depicts 4, label is 1 otherwise
        y[y == binary[0]] = 0
        y[y == binary[1]] = 1
    return x, y


def load_pendigits(split, random_state, binary=[], model='other'):
    """
    [Load and split the data]
    
    Args:
        split ([type]): [description]
        random_state ([int]): [Random seed used to initialize the pseudo-random number generator.] Defaults to 0.
        binary ([[] or list of size 2]): [    If binary == [], then it loads all the classes.
        Otherwise it loads the samples belonging to the classes whose labels are in the list.]]    
    Returns:
        [type]: [description]
    """
    x, y = read_pendigits("dataloaders/data/pendigits", binary)
    return split_data(x, y, split, random_state, model=model)


### Iris
def load_iris(split, random_state, binary=[], model='other'):
    """
    [Load and split the data.
    Setosa class is encoded as 0.
    Versicolor is encoded as 1.
    Virginica is encode as 2.]
    
    Args:
        split ([type]): [description]
        random_state ([int]): [Random seed used to initialize the pseudo-random number generator.] Defaults to 0.
        binary ([[] or list of size 2]): [    If binary == [], then it loads all the classes.
        Otherwise it loads the samples belonging to the classes whose labels are in the list.]]    
    Returns:
        [type]: [description]
    """
    x, y = ds.load_iris(return_X_y=True)

    if (binary != []):
        condition = np.isin(y, binary)
        x = x[condition, :]
        y = y[condition]
        y[y == binary[0]] = 0
        y[y == binary[1]] = 1

    return split_data(x, y, split, random_state, model=model)


### Covertype
def load_covertype(split, random_state, binary=[], model='other'):
    """
    [Load and split the data.]
    
    Args:
        split ([type]): [description]
        random_state ([int]): [Random seed used to initialize the pseudo-random number generator.] Defaults to 0.
        binary ([[] or list of size 2]): [    If binary == [], then it loads all the classes.
        Otherwise it loads the samples belonging to the classes whose labels are in the list.]]    
    Returns:
        [type]: [description]
    """
    x, y = ds.fetch_covtype(data_home="dataloaders/data/Covertype",
                            return_X_y=True)
    if (binary != []):
        condition = np.isin(y, binary)
        x = x[condition, :]
        y = y[condition]
        y[y == binary[0]] = 0
        y[y == binary[1]] = 1
    return split_data(x, y, split, random_state, model=model)


### Voices
def load_voices(split, random_state, binary=[], model='other'):

    df = pd.read_csv('dataloaders/data/gender_voice_dataset.csv', sep=',')
    voice_data = df.to_numpy()
    y = LabelEncoder().fit_transform(voice_data[:, -1])
    x = np.array(voice_data[:, :-1], dtype=float)

    return split_data(x, y, split, random_state, model=model)


def load_diabetes(split, random_state, binary=[], model='other'):

    df = pd.read_csv('dataloaders/data/diabetes.csv', sep=',')
    diabetes_data = df.to_numpy()
    y = diabetes_data[:, -1]
    x = np.array(diabetes_data[:, :-1], dtype=float)

    return split_data(x, y, split, random_state, model=model)


def mnist_fashion(split, random_state, binary=[], model='other'):

    x = np.load("dataloaders/data/samples_fashion.npy")
    y = np.load("dataloaders/data/labels_fashion.npy")

    return split_data(x, y, split, random_state, model=model)


def mnist(split, random_state, binary=[], model='other'):

    x = np.load("dataloaders/data/mn_samples.npy")
    y = np.load("dataloaders/data/mn_labels.npy")

    return split_data(x, y, split, random_state, model=model)


def load_20news(split, random_state, binary=[], model='other'):
    """       
    The names of the target categories can be loaded as datasets.fetch_20newsgroups_vectorized()['target_names']:
   [('alt.atheism', 0),
 ('comp.graphics', 1),
 ('comp.os.ms-windows.misc', 2),
 ('comp.sys.ibm.pc.hardware', 3),
 ('comp.sys.mac.hardware', 4),
 ('comp.windows.x', 5),
 ('misc.forsale', 6),
 ('rec.autos', 7),
 ('rec.motorcycles', 8),
 ('rec.sport.baseball', 9),
 ('rec.sport.hockey', 10),
 ('sci.crypt', 11),
 ('sci.electronics', 12),
 ('sci.med', 13),
 ('sci.space', 14),
 ('soc.religion.christian', 15),
 ('talk.politics.guns', 16),
 ('talk.politics.mideast', 17),
 ('talk.politics.misc', 18),
 ('talk.religion.misc', 19)]
    [Load and split the data.]
    
    Args:
        split ([type]): [description]
        random_state ([int]): [Random seed used to initialize the pseudo-random number generator.] Defaults to 0.
        binary ([[] or list of size 2]): [    If binary == [], then it loads all the classes.
        Otherwise it loads the samples belonging to the classes whose labels are in the list.]]    
    Returns:
        [type]: [description]
    """
    Features_data20news = fetch_20newsgroups()['data']
    Targets = fetch_20newsgroups()['target']
    vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, min_df=0.01)
    Features = vectorizer.fit_transform(Features_data20news)
    x = Features.todense().view(type=np.ndarray)
    y = Targets.astype(np.int)
    if binary != []:
        # classification task is to distinguish between 4 and 9
        condition = np.isin(y, binary)
        x = x[condition, :]
        y = y[condition]
        # label is 0, when the image depicts 4, label is 1 otherwise
        y[y == binary[0]] = 0
        y[y == binary[1]] = 1

    return split_data(x, y, split, random_state, model=model)

def load_soccer(split, random_state, binary=[], model = 'other'):
    
    x = np.load("dataloaders/data/soccer_features.npy")
    y = np.load("dataloaders/data/soccer_labels.npy")

    if (binary != []):
        condition = np.isin(y, binary)
        x = x[condition, :]
        y = y[condition]
        y[y == binary[0]] = 0
        y[y == binary[1]] = 1
    return split_data(x, y, split, random_state, model = model)

def load_soccer_filt1(split, random_state, binary=[], model = 'other'):
    
    x = np.load("dataloaders/data/soccer_features_filt1.npy")
    y = np.load("dataloaders/data/soccer_labels_filt1.npy")

    return split_data(x, y, split, random_state, model = model)

def load_soccer_filt2(split, random_state, binary=[], model = 'other'):
    
    x = np.load("dataloaders/data/soccer_features_filt2.npy")
    y = np.load("dataloaders/data/soccer_labels_filt2.npy")

    return split_data(x, y, split, random_state, model = model)

def load_soccer_filt3(split, random_state, binary=[], model = 'other'):
    
    x = np.load("dataloaders/data/soccer_features_filt3.npy")
    y = np.load("dataloaders/data/soccer_labels_filt3.npy")

    return split_data(x, y, split, random_state, model = model)

def load_AirlineTweets(split, random_state, binary=[], model='other'):
    """       
    Target categories: 0 - negative, 1 - neutral, 2 - positive
   
    Args:
        split ([type]): [description]
        random_state ([int]): [Random seed used to initialize the pseudo-random number generator.] Defaults to 0.
        binary ([[] or list of size 2]): [    If binary == [], then it loads all the classes.
        Otherwise it loads the samples belonging to the classes whose labels are in the list.]]    
    Returns:
        [type]: [description]
    """
    x = sparse.load_npz(
        "dataloaders/data/AirlineTweetsFeaturesSparse.npz").todense()
    y = np.load("dataloaders/data/AirlineTweetsTargetsArray.npy")
    if binary != []:
        # classification task is to distinguish between 4 and 9
        condition = np.isin(y, binary)
        x = x[condition, :]
        y = y[condition]
        # label is 0, when the image depicts 4, label is 1 otherwise
        y[y == binary[0]] = 0
        y[y == binary[1]] = 1

    return split_data(x, y, split, random_state, model=model)


def load_SMS(split, random_state, model='other'):
    """
    [Load and split the data.]

        random_state ([int]): [Random seed used to initialize the pseudo-random number generator.] Defaults to 0.]    
    Returns:
        [type]: [description]
    """
    x = np.load('dataloaders/data/SMS_spam_ham_features.npy',
                allow_pickle=True)
    y = np.load('dataloaders/data/SMS_spam_ham_targets.npy', allow_pickle=True)

    return split_data(x, y, split, random_state, model=model)


### Cifar-10
def load_cifar10(split, random_state, binary=[], model='other', representation="raw"):
    """
    [Load and split the data. Available labels are Auto(1), Bird(2), Truck(9)]
    
    Args:
        split ([type]): [description]
        random_state ([int]): [Random seed used to initialize the pseudo-random number generator.] Defaults to 0.]    
    Returns:
        [type]: [description]
    """
    if representation == "raw":
        x = np.load("dataloaders/data/cifar10/vec_features.npy")
    elif representation == "FFT2":
        x = np.load("dataloaders/data/cifar10/FFT2_features.npy")
    elif representation == "DWT2":
        x = np.load("dataloaders/data/cifar10/DWT2_features.npy")

    y = np.uint8(np.load("dataloaders/data/cifar10/vec_labels.npy"))

    if (binary != []):
        condition = np.isin(y, binary)
        x = x[condition, :]
        y = y[condition]
        y[y == binary[0]] = 0
        y[y == binary[1]] = 1

    return split_data(x, y, split, random_state, model=model)


### STL-10
def load_STL10(split, random_state, binary=[], model='other', representation="raw"):
    """
    [Load and split the data. Available labels are Auto(1), Bird(2), Truck(9)]
    
    Args:
        split ([type]): [description]
        random_state ([type]): [description]
        binary (list, optional): [description]. Defaults to [].
        model (str, optional): [description]. Defaults to 'other'.
        representation (str, optional): [description]. Defaults to "raw".
    """
    if representation == "raw":
        x = np.load("dataloaders/data/STL10/vec_features.npy")
    elif representation == "FFT2":
        x = np.load("dataloaders/data/STL10/FFT2_features.npy")
    elif representation == "DWT2":
        x = np.load("dataloaders/data/STL10/DWT2_features.npy")

    y = np.load("dataloaders/data/STL10/vec_labels.npy")

    if (binary != []):
        condition = np.isin(y, binary)
        x = x[condition, :]
        y = y[condition]
        y[y == binary[0]] = 0
        y[y == binary[1]] = 1

    return split_data(x, y, split, random_state, model=model)


def load_mushrooms(split, random_state, model='other'):
    """
    [Load and split the data.]
    
    Args:
        split ([type]): [description]
        random_state ([int]): [Random seed used to initialize the pseudo-random number generator.] Defaults to 0.]    
    Returns:
        [type]: [description]
    """
    x = np.load('dataloaders/data/Mushrooms_features.npy')
    y = np.load('dataloaders/data/Mushrooms_targets.npy')

    return split_data(x, y, split, random_state, model=model)
