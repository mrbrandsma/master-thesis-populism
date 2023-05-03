from helper_features import get_features, get_tfidf, create_vectorizer, dataframe_coefficients
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm
from sklearn.metrics import classification_report
import seaborn as sns
from nltk.corpus import stopwords

def data_processing(dataset,
                    target_label,
                    dev=True):
    """
    - imports the wanted datasets
    - Extracts the list of instances
    - Extracts the label
    Returns a list of instances for the training and dev/test set, a list of labels for the training and 
    dev/test set
    """
    tweets_train = []
    tweets_test = []
    labels_train = []
    labels_test = []

    # Import the training set
    filepath_train = (f'../data/{dataset}/clean/clean_train.csv')
    df_tweets_train = pd.read_csv(filepath_train, sep=',')

    print("df_tweets_train: ", len(df_tweets_train))
    # Import the test set
    if dev is True:
        filepath_test = (f'../data/{dataset}/clean/clean_dev.csv')
    else:
        filepath_test = (f'../data/{dataset}/clean/clean_test.csv')
    df_tweets_test = pd.read_csv(filepath_test, sep=',')

    # Store tweets and gold labels of training set
    for tweet in df_tweets_train['Tweet']:
        tweets_train.append(tweet)
    for label in df_tweets_train[target_label]:
        labels_train.append(label)

    # Store tweets and gold labels of test set
    for tweet in df_tweets_test['Tweet']:
        tweets_test.append(tweet)
    for label in df_tweets_test[target_label]:
        labels_test.append(label)

    # Print information
    print('Length of training dataset: ', len(tweets_train))
    print('Length of testing dataset: ', len(tweets_test))

    return (tweets_train, labels_train, tweets_test, labels_test)


def numerical_labels(training_labels, test_labels):
    """
    Change labels from string to numerical values.
    """
    label_encoder = preprocessing.LabelEncoder()
    
    # Collect all labels from training set and test set
    label_encoder.fit(training_labels+test_labels)
    
    # Change to numerical labels
    training_classes = label_encoder.transform(training_labels)
    test_classes = label_encoder.transform(test_labels)

    return (training_classes, test_classes, label_encoder)


def svm_classifier(vector, classes, max_iterations):
    """
    A short function that creates a classifier that predicts labels of
    the test data. Returns this classifier.
    """
    svm_linear_clf = svm.LinearSVC(max_iter=max_iterations)
    svm_linear_clf.fit(vector, classes)

    return svm_linear_clf


def evaluator(test_classes, test_prediction, label_encoder, test_instances, dataset, outfile):
    """
    Calculate the scores of the model and write them to file
    """ 
    # Open file to write evaluation statistics
    f = open(f'../results/{dataset}/descriptives_{outfile}.txt', 'a', encoding="utf-8")
    f.write('\n--------------EVALUATION---------------\n')

    # Calculate all statistics
    # #micro_recall = sklearn.metrics.recall_score(y_true=test_classes, 
    #                                             y_pred=test_prediction, 
    #                                             average='micro')
    # #macro_recall = sklearn.metrics.recall_score(y_true=test_classes, 
    #                                             y_pred=test_prediction, 
    #                                             average='macro')
    # #micro_precision = sklearn.metrics.precision_score(y_true=test_classes, 
    #                                                   y_pred=test_prediction, 
    #                                                   average='micro')
    # #macro_precision = sklearn.metrics.precision_score(y_true=test_classes, 
    #                                                   y_pred=test_prediction, 
    #                                                   average='macro')
    # #micro_f = sklearn.metrics.f1_score(y_true=test_classes, 
    #                                    y_pred=test_prediction, 
    #                                    average='micro')
    # #macro_f = sklearn.metrics.f1_score(y_true=test_classes, 
    #                                    y_pred=test_prediction, 
    #                                    average='macro')

    # Write the data to file
    #f.write(f'Micro recall: {micro_recall}\n')
    #f.write(f'Macro recall: {macro_recall}\n')
    #f.write(f'Micro precision: {micro_precision}\n')
    #f.write(f'Macro precision: {macro_precision}\n')
    #f.write(f'Micro f-score: {micro_f}\n')
    #f.write(f'Macro f-score: {micro_f}\n')

    # Get the classification report
    report = classification_report(test_classes, test_prediction, digits = 7)
    f.write('\nClassification report: \n')
    f.write(f'{label_encoder.classes_}\n')
    f.write(f'{report}\n')

    # Print to terminal as well
    print(f'{label_encoder.classes_}')
    print(f'{report}')

    # Get the confusion matrix
    cf_matrix = sklearn.metrics.confusion_matrix(test_classes, test_prediction)
    f.write(f'Confusion matrix: \n')
    f.write(f'{label_encoder.classes_}\n')
    f.write(f'{cf_matrix}\n')

    # Create heatmap
    sns.heatmap(cf_matrix, annot=True, fmt='.2%', cmap='Blues')
    plt.savefig(f'../results/{dataset}/heatmap.png')
    plt.clf()

    # Store twenty different cases
    f.write('\nSentence check:\n')
    f.write(f'{label_encoder.classes_}\n')
    f.write('Instance, true label, predicted label\n')
    
    instance = 0
    for i in range(20):
        instance += 130
        f.write(f'{instance}. "{test_instances[instance]}": {test_classes[instance]}, {test_prediction[instance]}\n')

    # Close file
    f.write('---------------------------------------\n')
    f.close()


def classifier(data_set,
               target_label,
               min_frequency,
               max_frequency,
               iterations,
               feature_dict,
               stopwords,
               dev,
               outfile
              ):
    """
    Main function that creates the classifier and evaluates it.
    """
    # Import the data
    tweets_train, labels_train, tweets_test, labels_test = data_processing(data_set, target_label, dev=dev)

    #assert len(tweets_train) == 21673, "Length of training dataset is incorrect"
    #assert len(labels_train) == 21673, "Length of training labels is incorrect"
    #if dev == False:
    #    assert len(tweets_test) == 4632, "Length of testing dataset is incorrect"
    #    assert len(labels_test) == 4632, "Length of testing labels is incorrect" 
    
    # Get TF-IDF vectors
    if feature_dict['tf-idf'] == True:
        train_tfidf_vectors, test_tfidf_vectors, vec = get_tfidf(tweets_train, tweets_test, min_frequency, stopwords)
        print("Length of train_tfidf_vectors", len(train_tfidf_vectors))
        print("Length of test_tfidf_vectors", len(test_tfidf_vectors))

    print("Getting training features")
    features_train = get_features(tweets_train,  feature_dict)
    features_test = get_features(tweets_test, feature_dict)

    print("Vectorizing features")
    # Vectorize the features
    if feature_dict['tf-idf'] == False:
        vec, features_vector_train = create_vectorizer(features_train)
        features_vector_test = vec.transform(features_test)

        print("Vectors length: ", features_vector_train.shape, features_vector_test.shape)

    print("Making labels numerical")
    # Make labels numerical
    training_classes, test_classes, label_encoder = numerical_labels(labels_train, labels_test)

    # Train model
    print("Getting trained model")
    if feature_dict['tf-idf'] == False:
        trained_svm = svm_classifier(features_vector_train, training_classes, iterations)

        # Predict labels for test set
        print("Predicting labels for test set")
        test_predict = trained_svm.predict(features_vector_test)
    else:
        trained_svm = svm_classifier(train_tfidf_vectors, training_classes, iterations)
        # Predict labels for test set
        print("Predicting labels for test set with TF-IDF")
        test_predict = trained_svm.predict(test_tfidf_vectors)
    
    # Get statistics
    if feature_dict['tf-idf'] == True:
        a, n, p = dataframe_coefficients(trained_svm, vec, data_set, outfile, top_features=10)

    # Test the model
    print("Testing the model")
    evaluator(test_classes, test_predict, label_encoder, tweets_test, data_set, outfile)