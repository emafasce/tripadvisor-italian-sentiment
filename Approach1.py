from sklearn.feature_extraction.text import TfidfVectorizer
import treetaggerwrapper
import string
import csv
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
    
def tokenizer(s):
    for punct in string.punctuation:
        s=s.replace(punct, " ")
    tags = tagger.tag_text(s)
    return [(x.split("\t"))[2] for x in tags if (x.split("\t"))[1]]

#Reading files
    
train_file="development.csv"
text_file="evaluation.csv"

reviews=[]
grade=[]
tagger = treetaggerwrapper.TreeTagger(TAGLANG='it')

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
        
#Building the Tf-Idf matrix

vectorizer=TfidfVectorizer(encoding='utf-8', strip_accents='unicode', lowercase=True,
                           tokenizer=tokenizer, stop_words=None, ngram_range=(1, 2))

print("Starting X matrix fitting...")
X=vectorizer.fit_transform(reviews)

#Splitting into training test and test set

Xtr=X[0:28754]
Xte=X[28754:]

#Building the model

model2=Sequential()
model2.add(Dense(input_dim=Xtr.shape[1], units=128, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(units=64, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(units=1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer=Adam())
callback=EarlyStopping(monitor='val_loss', patience=0)
model2.summary()

#Creating validation set

XTrain, XTest, yTrain, yTest = train_test_split(Xtr, grade,test_size = 0.2)

#Fitting the model

history = model2.fit(XTrain, yTrain, epochs=2, verbose=True, workers=-1, validation_data=(XTest, yTest), callbacks=[callback])

#Storing predictions into ypred

ypred=model2.predict(Xte)

#Transforming into binary predictions

ypred2=[]
for y in ypred:
    if abs(y-1)<abs(y):
        ypred2.append(1)
    else:
        ypred2.append(0)

#Saving to file

with open('solutionsneural.csv', mode='w') as f:
    employee_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(['Id', 'Predicted'])
    for i,j in enumerate(ypred2):
        if j==1:
            employee_writer.writerow([i,"pos"])
        else:
            employee_writer.writerow([i,"neg"])
