
# coding: utf-8

# In[18]:


#coding:utf-8


#load data
train_texts = open('train_x.txt',encoding = 'utf-8' ).read().split('\n')
train_labels = open('train_y.txt').read().split('\n')
test_texts = open('test_x.txt',encoding='utf-8').read().split('\n')
test_labels = open('test_y.txt').read().split('\n')
all_text = train_texts + test_texts


#feature extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer   
count_v0= CountVectorizer();  
counts_all = count_v0.fit_transform(all_text);
count_v1= CountVectorizer(vocabulary=count_v0.vocabulary_);  
counts_train = count_v1.fit_transform(train_texts);   
print ("the shape of train is "+repr(counts_train.shape))  
count_v2 = CountVectorizer(vocabulary=count_v0.vocabulary_);  
counts_test = count_v2.fit_transform(test_texts);  
print ("the shape of test is "+repr(counts_test.shape)) 
tfidftransformer = TfidfTransformer();    
train_data = tfidftransformer.fit(counts_train).transform(counts_train);
test_data = tfidftransformer.fit(counts_test).transform(counts_test); 
x_train = train_data
y_train = train_labels
x_test = test_data
y_test = test_labels

#classifier
from sklearn.naive_bayes import MultinomialNB  
from sklearn import metrics
clf = MultinomialNB(alpha = 1)   

clf.fit(x_train, y_train);  
preds = clf.predict(x_test);
num = 0
preds = preds.tolist()
for i,pred in enumerate(preds):
    if int(pred) == int(y_test[i]):
        num += 1
print ('Accuracy score:' + str(float(num) / len(preds)))





        




