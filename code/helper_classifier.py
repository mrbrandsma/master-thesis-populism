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
    svm_linear_clf = svm.LinearSVC(max_iter=max_iterations, dual=False)
    svm_linear_clf.fit(vector, classes)

    return svm_linear_clf


def evaluator(test_classes, test_prediction, label_encoder, test_instances, dataset, outfile):
    """
    Calculate the scores of the model and write them to file
    """ 
    # Open file to write evaluation statistics
    f = open(f'../results/{dataset}/descriptives_{outfile}.txt', 'a', encoding="utf-8")
    f.write('\n--------------EVALUATION---------------\n')

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
    plt.savefig(f'../results/{dataset}/heatmap_{outfile}.png')
    plt.clf()

    # Get the error analysis
    error_analysis(test_instances, test_classes, test_prediction, dataset, outfile, label_encoder)

    # Close file
    f.write('---------------------------------------\n')
    f.close()


def error_analysis(tweet, gold_label, prediction, dataset, outfile, label_encoder):
    """
    Prints a certain amount of mistakes and a certain amount of good predictions.
    """
    # Open file to write to
    f = open(f'../results/{dataset}/descriptives_{outfile}.txt', 'a', encoding="utf-8")
    f.write('\n--------------ERROR ANALYSIS---------------\n')

    # Change the amount of mistakes printed here
    amount_of_mistakes = int(40 / 4)

    # Merge all data together
    data_together = zip(tweet, gold_label, prediction)
    labels = ['Tweet', 'Gold label', 'Prediction']
    predictions_dataframe = pd.DataFrame(data_together, columns = labels)

    f.write("MISTAKES:\n")
    
    labels = label_encoder.classes_

    # Order examples
    populist_examples = predictions_dataframe.loc[predictions_dataframe['Gold label'] == 0]
    non_populist_examples = predictions_dataframe.loc[predictions_dataframe['Gold label'] == 1]
    
    f.write(f">{labels[0]} tweets predicted as {labels[-1]} tweets:\n")
    wrong_populist = populist_examples.loc[populist_examples['Prediction'] == 1]
    print_score_string = wrong_populist.to_string(header=True, index=False, max_rows=amount_of_mistakes, max_cols=1)
    f.write(print_score_string)
    f.write('\n\n')

    f.write(f">{labels[-1]} tweets predicted as {labels[0]} tweets:\n")
    wrong_non_populist = non_populist_examples.loc[non_populist_examples['Prediction'] == 0]
    print_score_string = wrong_non_populist.to_string(header=True, index=False, max_rows=amount_of_mistakes, max_cols=1)
    f.write(print_score_string)
    f.write('\n\n')

    f.write("CORRECT:\n")
    f.write(f">{labels[0]} tweets predicted as {labels[0]} tweets:\n")
    correct_populist = populist_examples.loc[populist_examples['Prediction'] == 0]
    print_score_string = correct_populist.to_string(header=True, index=False, max_rows=amount_of_mistakes, max_cols=1)
    f.write(print_score_string)
    f.write('\n\n')

    f.write(f">{labels[-1]} tweets predicted as {labels[-1]} tweets:\n")
    correct_non_populist = non_populist_examples.loc[non_populist_examples['Prediction'] == 1]
    print_score_string = correct_non_populist.to_string(header=True, index=False, max_rows=amount_of_mistakes, max_cols=1)
    f.write(print_score_string)
    f.write('\n\n')


def classifier(data_set,
               target_label,
               min_frequency,
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
    
    # Get TF-IDF vectors
    if feature_dict['tf-idf'] == True:
        train_tfidf_vectors, test_tfidf_vectors, vec = get_tfidf(tweets_train, tweets_test, min_frequency, stopwords)

    print("Getting training features")
    features_train = get_features(tweets_train, feature_dict)
    features_test = get_features(tweets_test, feature_dict)

    print("Vectorizing features")
    # Vectorize the features
    if feature_dict['tf-idf'] == False:
        vec, features_vector_train = create_vectorizer(features_train)
        features_vector_test = vec.transform(features_test)

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
        a, n, p = dataframe_coefficients(trained_svm, vec, data_set, outfile, label_encoder, top_features=10)

    # Test the model
    print("Testing the model")
    evaluator(test_classes, test_predict, label_encoder, tweets_test, data_set, outfile)