#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORT LIBRARY

import pandas as pd
import numpy as np
import nltk
import string
import re
import csv
import ast
import sklearn
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn import svm


# In[2]:


dataset = pd.read_csv('HasilLexiconRevisi.csv')


# In[3]:


dataset.head()


# In[4]:


dataset


# In[5]:


sentimen = []
for index, row in dataset.iterrows():
    if row["Label Tweet"] > 0:
        sentimen.append("Positif")
    elif row["Label Tweet"] < 0:
        sentimen.append("Negatif")
    else:
        sentimen.append("Netral")


# In[6]:


dataset['Label Tweet'] = sentimen


# In[7]:


dataset.sort_values('Label Tweet', ascending = True)


# In[8]:


dataset['Label Tweet'].value_counts()


# In[9]:


dataset['Label Tweet'].to_csv('convert.csv', index = False)


# In[160]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset['Tweet Akhir'], dataset['Label Tweet'], test_size=0.2, random_state=36)


# In[161]:


df_train = pd.DataFrame()
df_train['Tweet Akhir'] = X_train
df_train['Label Tweet'] = y_train

df_test = pd.DataFrame()
df_test['Tweet Akhir'] = X_test
df_test['Label Tweet'] = y_test


# In[162]:


df_train


# In[163]:


df_test


# In[164]:


df_train['Label Tweet'].value_counts()


# In[165]:


df_test['Label Tweet'].value_counts()


# #PROSES PEMBOBOTAN TF-IDF

# In[166]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(max_features = 5000)
tfidf_vect.fit(dataset['Tweet Akhir'])
train_X_tfidf = tfidf_vect.transform(df_train['Tweet Akhir'])
test_X_tfidf = tfidf_vect.transform(df_test['Tweet Akhir'])


# In[167]:


tfidf_vect


# In[168]:


print(train_X_tfidf)


# In[169]:


print(test_X_tfidf)


# In[170]:


print(train_X_tfidf.shape)
print(test_X_tfidf.shape)


# In[171]:


print(tfidf_vect.vocabulary_)


# In[172]:


train_X_tfidf.todense()


# In[173]:


df_trainmatrix = pd.DataFrame(train_X_tfidf.todense().T,
                 index = tfidf_vect.get_feature_names(),
                 columns=[f'D{i+1}' for i in range(len(df_train['Tweet Akhir']))])


# In[174]:


df_trainmatrix


# In[175]:


test_X_tfidf.todense()


# In[176]:


df_testmatrix = pd.DataFrame(test_X_tfidf.todense().T,
                 index = tfidf_vect.get_feature_names(),
                 columns=[f'D{i+1}' for i in range(len(df_test['Tweet Akhir']))])


# In[177]:


df_testmatrix


# #DEKLARASI GRIDSEARCHCV

# In[178]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier


# In[179]:


model_to_set_linear = OneVsRestClassifier(SVC(kernel="linear"))
model_to_set_rbf = OneVsRestClassifier(SVC(kernel="rbf"))
model_to_set_poly = OneVsRestClassifier(SVC(kernel="poly"))
model_to_set_sigmoid = OneVsRestClassifier(SVC(kernel="sigmoid"))


# In[180]:


parameters_linear= {
    "estimator__C": [0.001, 0.01, 0.1, 1, 1.5, 2, 2.5, 10, 20, 100],
    "estimator__max_iter": [0.001, 0.01, 0.1, 1, 1.5, 2, 2.5, 10, 20, 100],
    "estimator__kernel":["linear"],
}


# In[181]:


parameters_rbf= {
    "estimator__C": [0.001, 0.01, 0.1, 1, 1.5, 2, 2.5, 10, 20, 100],
    "estimator__gamma": [0.01, 0.1, 0.5, 1, 10, 20, 40, 50, 100, 1000],
    "estimator__kernel":["rbf"],
}


# In[182]:


parameters_poly= {
    "estimator__C": [0.001, 0.01, 0.1, 1, 1.5, 2, 2.5, 10, 20, 100],
    "estimator__gamma": [0.01, 0.1, 0.5, 1, 10, 20, 40, 50, 100, 1000],
    "estimator__degree": [0.01, 0.1, 0.5, 1, 10, 20, 40, 50, 100, 1000],
    "estimator__coef0": [0, 1, 2, 3, 5, 8, 10, 20, 25, 50, 75, 100],
    "estimator__kernel":["poly"],
}


# In[183]:


parameters_sigmoid= {
    "estimator__C": [0.001, 0.01, 0.1, 1, 1.5, 2, 2.5, 10, 20, 100],
    "estimator__gamma": [0.01, 0.1, 0.5, 1, 10, 20, 40, 50, 100, 1000],
    "estimator__coef0": [0, 1, 2, 3, 5, 8, 10, 20, 25, 50, 75, 100],
    "estimator__kernel":["sigmoid"],
}


# In[216]:


model_tunning_linear = GridSearchCV(model_to_set_linear, param_grid=parameters_linear,  cv=10, refit=True,verbose=3)
model_tunning_linear.fit(train_X_tfidf,df_train['Label Tweet'])


# In[217]:


model_tunning_rbf = GridSearchCV(model_to_set_rbf, param_grid=parameters_rbf,  cv=10, refit=True,verbose=3)
model_tunning_rbf.fit(train_X_tfidf,df_train['Label Tweet'])


# In[191]:


model_tunning_poly = GridSearchCV(model_to_set_poly, param_grid=parameters_poly,  cv=10, refit=True,verbose=3)
model_tunning_poly.fit(train_X_tfidf,df_train['Label Tweet'])


# In[192]:


model_tunning_sigmoid = GridSearchCV(model_to_set_sigmoid, param_grid=parameters_sigmoid,  cv=10, refit=True,verbose=3)
model_tunning_sigmoid.fit(train_X_tfidf,df_train['Label Tweet'])


# In[218]:


print(model_tunning_linear.best_params_)
print(model_tunning_rbf.best_params_)
print(model_tunning_poly.best_params_)
print(model_tunning_sigmoid.best_params_)


# In[219]:


model_tunning_linear_predictions = model_tunning_linear.predict(test_X_tfidf)
model_tunning_rbf_predictions = model_tunning_rbf.predict(test_X_tfidf)
model_tunning_poly_predictions = model_tunning_poly.predict(test_X_tfidf)
model_tunning_sigmoid_predictions = model_tunning_sigmoid.predict(test_X_tfidf)


# In[220]:


from sklearn.metrics import plot_confusion_matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(model_tunning_linear, test_X_tfidf, y_test,
#                                  display_labels=class_names,
                                 cmap=plt.cm.PuBuGn,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)

plt.show()


# In[221]:


from sklearn.metrics import plot_confusion_matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(model_tunning_rbf, test_X_tfidf, y_test,
#                                  display_labels=class_names,
                                 cmap=plt.cm.PuBuGn,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)

plt.show()


# In[222]:


from sklearn.metrics import plot_confusion_matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(model_tunning_poly, test_X_tfidf, y_test,
#                                  display_labels=class_names,
                                 cmap=plt.cm.PuBuGn,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)

plt.show()


# In[223]:


from sklearn.metrics import plot_confusion_matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(model_tunning_sigmoid, test_X_tfidf, y_test,
#                                  display_labels=class_names,
                                 cmap=plt.cm.PuBuGn,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)

plt.show()


# In[224]:


# clasifikasi report linear
from sklearn.metrics import classification_report
print(classification_report(y_test, model_tunning_linear_predictions))


# In[225]:


# clasifikasi report RBF
from sklearn.metrics import classification_report
print(classification_report(y_test, model_tunning_rbf_predictions))


# In[226]:


# clasifikasi report Polynomial
from sklearn.metrics import classification_report
print(classification_report(y_test, model_tunning_poly_predictions))


# In[227]:


# clasifikasi report Sigmoid
from sklearn.metrics import classification_report
print(classification_report(y_test, model_tunning_sigmoid_predictions))


# In[228]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[229]:


model_tunning_linear_accuracy = accuracy_score(y_test,model_tunning_linear_predictions)
model_tunning_linear_precision = precision_score(y_test,model_tunning_linear_predictions, average='macro')
model_tunning_linear_recall = recall_score(y_test,model_tunning_linear_predictions, average='macro')
model_tunning_linear_f1 = f1_score(y_test,model_tunning_linear_predictions, average='macro')
print('Accuracy (Linear Kernel): ', "%.2f" % (model_tunning_linear_accuracy*100))
print('Precision (Linear Kernel): ', "%.2f" % (model_tunning_linear_precision*100))
print('Recall (Linear Kernel): ', "%.2f" % (model_tunning_linear_recall*100))
print('F1 (Linear Kernel): ', "%.2f" % (model_tunning_linear_f1*100))


# In[230]:


model_tunning_rbf_accuracy = accuracy_score(y_test,model_tunning_rbf_predictions)
model_tunning_rbf_precision = precision_score(y_test,model_tunning_rbf_predictions, average='macro')
model_tunning_rbf_recall = recall_score(y_test,model_tunning_rbf_predictions, average='macro')
model_tunning_rbf_f1 = f1_score(y_test,model_tunning_rbf_predictions, average='macro')
print('Accuracy (Linear Kernel): ', "%.2f" % (model_tunning_rbf_accuracy*100))
print('Precision (Linear Kernel): ', "%.2f" % (model_tunning_rbf_precision*100))
print('Recall (Linear Kernel): ', "%.2f" % (model_tunning_rbf_recall*100))
print('F1 (Linear Kernel): ', "%.2f" % (model_tunning_rbf_f1*100))


# In[231]:


model_tunning_poly_accuracy = accuracy_score(y_test,model_tunning_poly_predictions)
model_tunning_poly_precision = precision_score(y_test,model_tunning_poly_predictions, average='macro')
model_tunning_poly_recall = recall_score(y_test,model_tunning_poly_predictions, average='macro')
model_tunning_poly_f1 = f1_score(y_test,model_tunning_poly_predictions, average='macro')
print('Accuracy (Linear Kernel): ', "%.2f" % (model_tunning_poly_accuracy*100))
print('Precision (Linear Kernel): ', "%.2f" % (model_tunning_poly_precision*100))
print('Recall (Linear Kernel): ', "%.2f" % (model_tunning_poly_recall*100))
print('F1 (Linear Kernel): ', "%.2f" % (model_tunning_poly_f1*100))


# In[232]:


model_tunning_sigmoid_accuracy = accuracy_score(y_test,model_tunning_sigmoid_predictions)
model_tunning_sigmoid_precision = precision_score(y_test,model_tunning_sigmoid_predictions, average='macro')
model_tunning_sigmoid_recall = recall_score(y_test,model_tunning_sigmoid_predictions, average='macro')
model_tunning_sigmoid_f1 = f1_score(y_test,model_tunning_sigmoid_predictions, average='macro')
print('Accuracy (Linear Kernel): ', "%.2f" % (model_tunning_sigmoid_accuracy*100))
print('Precision (Linear Kernel): ', "%.2f" % (model_tunning_sigmoid_precision*100))
print('Recall (Linear Kernel): ', "%.2f" % (model_tunning_sigmoid_recall*100))
print('F1 (Linear Kernel): ', "%.2f" % (model_tunning_sigmoid_f1*100))


# In[233]:


result_predictiontest = pd.DataFrame()
result_predictiontest['Tweet Akhir'] = X_test
result_predictiontest['Hasil Prediktif'] = model_tunning_linear_predictions


# In[234]:


result_predictiontest


# In[235]:


result_predictiontest['Hasil Aktual'] = df_test['Label Tweet']


# In[236]:


result_predictiontest


# In[237]:


result_predictiontest['Hasil Aktual'].value_counts()


# In[238]:


result_predictiontest['Hasil Prediktif'].value_counts()


# In[239]:


labels = ['Negatif', 'Netral', 'Positif']
Category4 = [33, 4, 93]
plt.bar(labels, Category4, tick_label=labels, width=0.5, color = ['steelblue', 'coral', 'seagreen'])
plt.xlabel('Label Data')
plt.ylabel('Jumlah Data')
plt.title('Diagram Bar pada Hasil Support Vector Machine')


# In[240]:


labels = ['Negatif', 'Netral', 'Positif']
CategoryData = [33, 12, 85]
plt.bar(labels, CategoryData, tick_label=labels, width=0.5, color = ['steelblue', 'coral', 'seagreen'])
plt.xlabel('Label Data')
plt.ylabel('Jumlah Data')
plt.title('Diagram Bar pada Data Test')


# In[241]:


color = ['coral', 'c']
plt.pie(Category4, labels=labels,startangle=90, shadow=True, autopct='%1.2f%%', explode=(0.1, 0.1, 0.1))
plt.title('Diagram Pie pada Hasil Support Vector Machine')
plt.legend()
plt.show()


# In[242]:


color = ['coral', 'c']
plt.pie(CategoryData, labels=labels,startangle=90, shadow=True, autopct='%1.2f%%', explode=(0.1, 0.1, 0.1))
plt.title('Diagram Pie pada Data Test')
plt.legend()
plt.show()


# In[ ]:




