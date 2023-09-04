#Import Relevant Libraries

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from wordcloud import WordCloud,STOPWORDS
from nltk.tokenize import RegexpTokenizer
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import svm, metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay

#Reading data from CSV file
senti_data=pd.read_csv('uk_pm.csv')

#Displaying first 10 rows and info about data
senti_data.head(5)
senti_data.shape
senti_data.info()

#Data Cleaning
Checking for missing values
senti_data.isnull().sum()

#Replacing missing values with NA
senti_data.fillna('', inplace=True)
senti_data.isnull().sum()


#Pre Processing Tweets
def rem_short(data1):
    data1=re.sub("’","'",data1)
    data1=re.sub("isn't",'is not',data1)
    data1=re.sub("he's",'he is',data1)
    data1=re.sub("wasn't",'was not',data1)
    data1=re.sub("there's",'there is',data1)
    data1=re.sub("couldn't",'could not',data1)
    data1=re.sub("won't",'will not',data1)
    data1=re.sub("they're",'they are',data1)
    data1=re.sub("she's",'she is',data1)
    data1=re.sub("There's",'there is',data1)
    data1=re.sub("wouldn't",'would not',data1)
    data1=re.sub("haven't",'have not',data1)
    data1=re.sub("That's",'That is',data1)
    data1=re.sub("you've",'you have',data1)
    data1=re.sub("He's",'He is',data1)
    data1=re.sub("what's",'what is',data1)
    data1=re.sub("weren't",'were not',data1)
    data1=re.sub("we're",'we are',data1)
    data1=re.sub("hasn't",'has not',data1)
    data1=re.sub("you'd",'you would',data1)
    data1=re.sub("shouldn't",'should not',data1)
    data1=re.sub("let's",'let us',data1)
    data1=re.sub("they've",'they have',data1)
    data1=re.sub("You'll",'You will',data1)
    data1=re.sub("i'm",'i am',data1)
    data1=re.sub("we've",'we have',data1)
    data1=re.sub("it's",'it is',data1)
    data1=re.sub("don't",'do not',data1)
    data1=re.sub("that's",'that is',data1)
    data1=re.sub("I'm",'I am',data1)
    data1=re.sub("it's",'it is',data1)
    data1=re.sub("she's",'she is',data1)
    data1=re.sub("he's",'he is',data1)
    data1=re.sub("I'm",'I am',data1)
    data1=re.sub("I'd",'I did',data1)
    data1=re.sub("he's",'he is',data1)
    data1=re.sub("there's",'there is',data1)
    return data1
data['text_clean'] = data['text_clean'].apply(lambda x:rem_short(x))

def sep_rem(data):
    text1 = re.sub("[^a-zA-Z0-9.' ]+", "", data)
    cleaned_text = re.sub('\s+', ' ', text1).strip()
    return cleaned_text
data['text_clean']= data['text_clean'].apply(lambda x: sep_rem(x))

#Showing difference between normal tweet and cleaned tweet
Print(data['text'][222])
Print(data['text_clean'][222])

#Finding sentiment using Roberta model
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment" 
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def get_sentimental_score(text):
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    sentiment_score = output[0][0].detach().numpy()
    sentiment_score = softmax(sentiment_score)
    return sentiment_score

def classify_sentiment(text: str) -> str:
    """ Classify Sentiment Based On Generated Score [Postive, Neutral & Negative] """
    sentiment_label = ''
    sentiment_score = get_sentimental_score(text)
    if sentiment_score[0] > sentiment_score[1] and sentiment_score[0] > sentiment_score[2]:
        sentiment_label = "NEGATIVE"
    elif sentiment_score[1] > sentiment_score[2]:
        sentiment_label = "NEUTRAL"
    else:
        sentiment_label = "POSITIVE"
    return sentiment_label

# Apply Sentiment Generator To A New Column In Parent Df
data['sentiment_generated'] = data['text_clean'].apply(classify_sentiment)

print(data.head(5))

#Finding and plotting count of each sentiment found
senti_count = data['sentiment_generated'].value_counts()
senti_count.plot(kind='bar')
plt.xlabel('Name')
plt.ylabel('Count')
plt.title('Count of Different Sentiment')

#Plotting word cloud for each sentiment
grouped = data.groupby('sentiment_generated')
stopwords = set(STOPWORDS)
# Generate word cloud for each value
for sentiment, group in grouped:
    text = ' '.join(group['text_clean'])
    wordcloud = WordCloud(stopwords = stopwords).generate(text)

    # Plot the word cloud
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {sentiment} Sentiment')
    plt.axis('off')
    plt.show()

#Saving processed dataset to CSV file for further testing and training
data.to_csv('SentiAnalysis_RishiSunak.csv', encoding='utf-8')

#Importing CSV file to a new df for model traingn and testing
model_data =pd.read_csv('SentiAnalysis_RishiSunak.csv')
print(model_data.head(5))

Assigning values to Sentiment obtained
def map_sentiment(nature):
    if nature == 'POSITIVE':
        return 1
    elif nature == 'NEGATIVE':
        return -1
    else:
        return 0
model_data['sentiment'] = model_data['sentiment_generated'].apply(lambda x:map_sentiment(x))

#MACHINE LEARNING THROUGH SUPPORT VECTOR MACHINE

#Reprocess the text and extract features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(model_data['text_clean'])
y = model_data['sentiment']

#Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

#Displays the shape or dimensions of the training and testing data
print("X Train " ,X_train.shape)
print("X Test " ,X_test.shape)
print("Y Train " ,y_train.shape)
print("Y Test " ,y_test.shape)

#Train the SVM model
svm = SVC(kernel='linear', probability=True)svm_model.fit(X_train, y_train)
t0 = time.time()
svm.fit(X_train, y_train)
svm_training_time=time.time()-t0

#Predict the sentiment on the test set
y_pred = svm.predict(X_test)


#Displaying results
print(classification_report(y_test, y_pred))
svm_report=classification_report(y_test, y_pred,output_dict=True)
svm_accuracy=metrics.accuracy_score(y_test, y_pred)
svm_precision=metrics.precision_score(y_test, y_pred, average='weighted')
svm_recall= metrics.recall_score(y_test, y_pred, average='weighted')
svm_f1_score= svm_report['macro avg']['f1-score']
fl_score.append(svm_f1_score)
precision.append(svm_precision)
recall.append(svm_recall)

print ("recall",svm_recall)
print ("precision:",svm_precision)
print ("accuracy:", svm_accuracy)
print('tranining time : ',svm_training_time)

#Plotting Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot();

#Plotting Accuracy and Loss graph
svm_accuracy=svm_accuracy*100
svm_accuracy=round(svm_accuracy, 2)  
svm_acc=[0,round(svm_accuracy, 2)]
svm_loss=100-svm_accuracy
svm_loss=[0,round(svm_loss, 0)]
text=('Accuracy = ',svm_accuracy,' Loss = ',svm_loss)
plt.plot(svm_acc, svm_loss)
plt.title("Accuracy And Loss ")
plt.xlabel("Accuracy")
plt.ylabel("Loss")
plt.text(0, 2, text, fontsize = 13,rotation=36)
plt.show()

#Plotting ROC Curve
def get_all_roc_coordinates(y_real, y_proba):
    tpr_list = [0]
    fpr_list = [0]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list

def plot_roc_curve(tpr, fpr, scatter = True, ax = None):
    if ax == None:
        plt.figure(figsize = (5, 5))
        ax = plt.axes()

    if scatter:
        sns.scatterplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = [0, 1], y = [0, 1], color = 'green', ax = ax)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

import seaborn as sns
plt.figure(figsize = (12, 8))
bins = [i/20 for i in range(20)] + [1]
roc_auc_ovr = {}
for i in range(len(classes_svm)):
    c = classes_svm[i]
    df_aux = y_test.copy()
    df_aux['sentiment'] = [1 if y == c else 0 for y in y_test]
    df_aux['prob'] = y_prob_svm[:, i]
    ax = plt.subplot(2, 3, i+1)
    tpr, fpr = get_all_roc_coordinates(df_aux['sentiment'], df_aux['prob'])
    plot_roc_curve(tpr, fpr, scatter = False, ax = ax)
    ax.set_title("ROC Curve")
    roc_auc_ovr[c] = roc_auc_score(df_aux['sentiment'], df_aux['prob'])  
plt.tight_layout()

#MACHINE LEARNING THROUGH RANDOM FOREST CLASSIFIER

#Preprocess the text and extract features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(model_data['text_clean'])
y = model_data['sentiment']

#Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

#Model Training
from sklearn.ensemble import RandomForestClassifier
classifier_rm = RandomForestClassifier(random_state=10)

t0 = time.time()
classifier_rm.fit(X_train, y_train)
rf_training_time=time.time()-t0

#Predicting Sentiment
y_pred_rf = classifier_rm.predict(X_test)

#Printing output
rf_report=classification_report(y_test, y_pred_rf,output_dict=True)
rf_accuracy=metrics.accuracy_score(y_test, y_pred_rf)
rf_precision=metrics.precision_score(y_test, y_pred_rf, average='weighted')
rf_recall= metrics.recall_score(y_test, y_pred_rf, average='weighted')
rf_f1_score= rf_report['macro avg']['f1-score']
fl_score.append(rf_f1_score)
precision.append(rf_precision)
recall.append(rf_recall)

print ("recall",rf_recall)
print ("precision:",rf_precision)
print ("accuracy:", rf_accuracy)
print('rf training time :',rf_training_time)

#Plotting Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

#Plotting Accuracy Loss Graph
rf_accuracy=rf_accuracy*100
rf_accuracy=round(rf_accuracy, 2) 
acc=[0,round(rf_accuracy, 2)]
rf_loss=100-rf_accuracy
rf_loss=[0,round(rf_loss, 0) ]
text='Accuracy = ',rf_accuracy,' Loss = ',rf_loss
plt.plot(acc, rf_loss)
plt.title("Accuracy And Loss ")
plt.xlabel("Accuracy")
plt.ylabel("Loss")
plt.text(3, 5, text, fontsize = 13,rotation=36)
plt.show()


#Plotting ROC Curve
score = roc_auc_score(y_test, y_prob_rf, multi_class='ovr')
import seaborn as sns
plt.figure(figsize = (12, 8))
bins = [i/20 for i in range(20)] + [1]
roc_auc_ovr = {}

for i in range(len(classes_rf)):
    c = classes_rf[i]
    df_aux = y_test.copy()
    df_aux['sentiment'] = [1 if y == c else 0 for y in y_test]
    df_aux['prob'] = y_prob_rf[:, i]
    ax = plt.subplot(2, 3, i+1)
    tpr, fpr = get_all_roc_coordinates(df_aux['sentiment'], df_aux['prob'])
    plot_roc_curve(tpr, fpr, scatter = False, ax = ax)
    ax.set_title("ROC Curve")
    roc_auc_ovr[c] = roc_auc_score(df_aux['sentiment'], df_aux['prob'])
plt.tight_layout()

#SVM and RF F1 Score Comparison
fig, ax = plt.subplots()
model = ['svm', 'rf']
accuracy = [fl_score[0], fl_score[1]]
bar_labels = ['SVM',  'RF']
bar_colors = ['tab:red', 'tab:blue']

ax.bar(model, accuracy, label=bar_labels, color=bar_colors)
ax.set_ylabel('accuracy')
ax.set_xlabel('model')
ax.set_title('F1 score of SVM and RF')
ax.legend(title='model',loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()

#MACHINE LEARNING THROUGH LSTM	

#Preprocess the text and extract features
X = model_data['text_clean']
y = model_data['sentiment']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Tokenize the tweets
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1

#Convert the tweets to sequences
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

#Pad the sequences to have the same length
max_length = 100  # maximum length of a tweet
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

#Model Training
t0 = time.time()
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_length))
model.add(LSTM(128, dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

#Compile the model
model.summary()
history=model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_training_time=time.time()-t0

#Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

#Testing Model
y_pred_lstm = model.predict(X_test).round()

#Displaying output
from sklearn.metrics import confusion_matrix,classification_report, precision_score,auc,precision_recall_curve
import seaborn as sns

print(classification_report(y_test, y_pred))
lstm_report=classification_report(y_test, y_pred,output_dict=True)
lstm_precision = lstm_report['macro avg']['precision']
lstm_accuracy = lstm_report['accuracy']
lstm_recall = lstm_report['macro avg']['recall']
lstm_f1_score= lstm_report['macro avg']['f1-score']
print('lstm training time :',lstm_training_time)

#Plotting Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot();

#Accuracy and Loss Graph
lstm_accuracy=lstm_accuracy*100
lstm_accuracy=round(lstm_accuracy, 2)
def plot_learning_curve(history,epochs):
    # plot training and validation accuracy 
    epoch_range = range(1,epochs+1)
    plt.plot(epoch_range,history.history['accuracy'])
    plt.plot(epoch_range,history.history['val_accuracy'])
    plt.title('training and validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train','Val'],loc='upper left')
    plt.show()

    # plot training and validation loss
    plt.plot(epoch_range,history.history['loss'])
    plt.plot(epoch_range,history.history['val_loss'])
    plt.title('training and validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train','Val'],loc='upper left')
    plt.show()
plot_learning_curve(history,10)

#Accuracy of SVM, LSTM, RF

fig, ax = plt.subplots()

model = ['svm', 'LSTM', 'rf']
accuracy = [svm_accuracy, lstm_accuracy, rf_accuracy]
bar_labels = ['SVM', 'LSTM', 'RF']
bar_colors = ['tab:red', 'tab:blue', 'tab:orange']

ax.bar(model, accuracy, label=bar_labels, color=bar_colors)

ax.set_ylabel('accuracy')
ax.set_xlabel('model')
ax.set_title('')
ax.legend(title='model',loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()

Time Complexity Comparison
fig, ax = plt.subplots()

model = ['svm', 'LSTM', 'rf']
accuracy = [svm_training_time, lstm_training_time, rf_training_time]
bar_labels = ['SVM', 'LSTM', 'RF']
bar_colors = ['tab:red', 'tab:blue', 'tab:orange']

ax.bar(model, accuracy, label=bar_labels, color=bar_colors)

ax.set_ylabel('Time in Seconds')
ax.set_xlabel('Model')
ax.set_title('')
ax.legend(title='model',loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
