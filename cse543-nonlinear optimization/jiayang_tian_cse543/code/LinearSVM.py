
# coding: utf-8

# In[5]:


#coding:utf-8
from sklearn.svm import SVC

train_texts = open('train_x.txt',encoding = 'utf-8').read().split('\n')
train_labels = open('train_y.txt').read().split('\n')
test_texts = open('test_x.txt',encoding = 'utf-8').read().split('\n')
test_labels = open('test_y.txt').read().split('\n')
all_text = train_texts + test_texts


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer   
count_v0= CountVectorizer();  
counts_all = count_v0.fit_transform(all_text);
count_v1= CountVectorizer(vocabulary=count_v0.vocabulary_);  
counts_train = count_v1.fit_transform(train_texts);   
count_v2 = CountVectorizer(vocabulary=count_v0.vocabulary_);  
counts_test = count_v2.fit_transform(test_texts);   
tfidftransformer = TfidfTransformer()
train_data = tfidftransformer.fit(counts_train).transform(counts_train);
test_data = tfidftransformer.fit(counts_test).transform(counts_test); 

x_train = train_data
y_train = train_labels
x_test = test_data
y_test = test_labels


svc = SVC(kernel = 'linear') 
svc.fit(x_train,y_train)  
preds = svclf.predict(x_test);  
num = 0
preds = preds.tolist()
for i,pred in enumerate(preds):
    if int(pred) == int(y_test[i]):
        num += 1
print ('Accuracy score:' + str(float(num) / len(preds)))

