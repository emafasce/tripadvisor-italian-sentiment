import string
import csv 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.model_selection import train_test_split
import treetaggerwrapper
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from sklearn.ensemble import RandomForestRegressor

def tokenizer(s):
    for punct in string.punctuation:
        s=s.replace(punct, " ")
    tags = tagger.tag_text(s)
    return [(x.split("\t"))[2] for x in tags]

#Reading files

train_file="development.csv"
text_file="evaluation.csv"

reviews=[]
grade=[]

with open(train_file,encoding="utf8") as csv_file:
    reader=csv.reader(csv_file)
    next(reader)
    for row in reader:
        reviews.append(row[0])
        if row[1]=='pos':
            grade.append(1)
        else:
            grade.append(0)
y=grade

with open(text_file,encoding="utf8") as csv_file:
    reader=csv.reader(csv_file)
    next(reader)
    for row in reader:
        reviews.append(row[0])

#Stemming the reviews
        
tagger = treetaggerwrapper.TreeTagger(TAGLANG='it')
stemmed_reviews=[]
i=0
for r in reviews:
    i=i+1
    rev=''
    for w in word_tokenize(r):
        for punct in string.punctuation:
            w=w.replace(punct, " ")
        tags = tagger.tag_text(w)
        for p in tags:
            type_word=p.split("\t")[1]
            if type_word!="NUM" and type_word!='ORD' and 'DET' not in type_word and type_word!="PUNC":
                rev=rev+p.split("\t")[2]+' '
    print(f'{i}', end=' ')
    stemmed_reviews.append(rev)


#Creating words dictionary

index=0
word_list={}
for rev in stemmed_reviews:
    words=rev.split(" ")
    for w in words:
        if w not in word_list:
            word_list[w]=index
            index+=1

#Embedding reviews into arrays
            
reviews_dic=[]
for rev in stemmed_reviews:
    rev_arr=[]
    words=rev.split(" ")
    for w in words:
        if w!='\n':
            num=word_list[w]
            rev_arr.append(num)
    reviews_dic.append(rev_arr)

#Padding reviews with zeros
    
reviews_pad=[]
for arr in reviews_dic:
    myarr=np.array(arr)
    zero=np.zeros(shape=(1700,))
    zero[:myarr.shape[0]]=myarr
    reviews_pad.append(zero)

#Creating X matrix with the arrays

matX=np.zeros(shape=(len(reviews_dic), 1700))
for i in range(len(reviews_dic)):
    matX[i,:]=reviews_pad[i].astype(int)
    
#Splitting into training set and test set
    
trainX=matX[0:28754].astype(int)
testX=matX[28754:].astype(int)

#Building validation set

XTrain, XVal, yTrain, yVal = train_test_split(trainX, grade, test_size = 0.2)

#Building Bidirectional LSTM 

batch_size=64
max_feat=len(word_list)
model = Sequential()
model.add(Embedding(max_feat, 128, input_length=1700))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.fit(XTrain, yTrain,batch_size=batch_size, epochs=1, validation_data=(XVal, yVal))
ypred=model.predict(testX)

#Now all the results of the RNN are stored in ypred

#Now I build the Random Forest Regressor, starting with the Tf-Idf matrix

vectorizer=TfidfVectorizer(encoding='utf-8', strip_accents='unicode', lowercase=True,
                           tokenizer=tokenizer, stop_words=None)

X=vectorizer.fit_transform(reviews)

#Splitting into training set and test set

trX=X[0:28754]
teX=X[28754:]

#Building the random forest regressor model and predicting the results

rfr=RandomForestRegressor(n_estimators=200)
rfr.fit(trX, grade)
rfrpred=rfr.predict(teX)

#Storing the sum of the results into ypred_combined

ypred_combined=[]
for i, j in enumerate(rfrpred):
    ypred_combined.append(j+ypred[i])

#Storing the binary predictions into yfinal, the cutoff is 1
    
yfinal=[]
for y in ypred_combined:
    if y>1:
        yfinal.append(1)
    else:
        yfinal.append(0)
    
#Saving to file

with open('solutions_comb.csv', mode='w') as f:
    employee_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(['Id', 'Predicted'])
    for i,j in enumerate(yfinal):
        if j==1:
            employee_writer.writerow([i,"pos"])
        else:
            employee_writer.writerow([i,"neg"])