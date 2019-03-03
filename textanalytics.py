# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 12:57:05 2019

@author: Amy
"""

import glob
import nltk
import unicodedata
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk import pos_tag_sents
from nltk.corpus import stopwords
import re
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from nltk.tag.perceptron import PerceptronTagger
tagger=PerceptronTagger()
import ast
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

path='C:\\Users\\Amy\\Downloads\\all1314text\\*.txt'
files=glob.glob(path)
text=''

for document in files:
    with open(document,'r',errors='ignore') as single_file:
        read_file=single_file.read().replace('\n','')
    text=text+ ' '+read_file
    
file=open('C:\\Users\\Amy\\Downloads\\all1314text\\corpus.text','w')
file.write(text)
file.close()

with open('C:\\Users\\Amy\\Downloads\\all1314text\\corpus.text','r') as corpus:
    read_file=corpus.read()
sentences=sent_tokenize(read_file)

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

entiresentencedf = pd.DataFrame(sentences)
df_clean = pd.DataFrame()
df_clean['sentences']=entiresentencedf[0].apply(remove_accented_chars)

df_clean.to_csv('backupsentences.csv')

df_clean_copy=df_clean.copy()
#percentage pattern
# %, percentage,percent,percentile,percentage points
#numerical or word version
#word version - 1-9, tenth-digit
percentage_num=[]
percentage_word=[]
for index,row in df_clean.iterrows():
    r1 = re.findall(r"['-]?\d+\.?\d+\s?%",row['sentences'])
    r2= re.findall(r" ['-]?\d+\.?\d+\spercent",row['sentences'])
    r3= re.findall(r" ['-]?\d+\.?\d+\spercentile",row['sentences'])
    r4= re.findall(r" ['-]?\d+\.?\d+\spercentage",row['sentences'])
    r5= re.findall(r" ['-]?\d+\.?\d+\spercentage points",row['sentences'])
    #digits
    n1=re.findall(r"(?:f(?:ive|our)|s(?:even|ix)|t(?:hree|wo)|(?:ni|o)ne|eight|zero)\spercent",row['sentences'])
    n2=re.findall(r"(?:f(?:ive|our)|s(?:even|ix)|t(?:hree|wo)|(?:ni|o)ne|eight|zero)\spercentage",row['sentences'])
    n3=re.findall(r"(?:f(?:ive|our)|s(?:even|ix)|t(?:hree|wo)|(?:ni|o)ne|eight|zero)\spercentile",row['sentences'])
    n4=re.findall(r"(?:f(?:ive|our)|s(?:even|ix)|t(?:hree|wo)|(?:ni|o)ne|eight|zero)\spercentage points",row['sentences'])
    #in tenth-digits
    n5=re.findall(r"(?:fif|six|eigh|nine|(?:tw|sev)en|(?:thi|fo)r)ty['-]?(?:f(?:ive|our)|s(?:even|ix)|t(?:hree|wo)|(?:ni|o)ne|eight)\spercent",row['sentences'])
    n6=re.findall(r"(?:fif|six|eigh|nine|(?:tw|sev)en|(?:thi|fo)r)ty['-]?(?:f(?:ive|our)|s(?:even|ix)|t(?:hree|wo)|(?:ni|o)ne|eight)\spercentage",row['sentences'])
    n7=re.findall(r"(?:fif|six|eigh|nine|(?:tw|sev)en|(?:thi|fo)r)ty['-]?(?:f(?:ive|our)|s(?:even|ix)|t(?:hree|wo)|(?:ni|o)ne|eight)\spercentile",row['sentences'])
    n8=re.findall(r"(?:fif|six|eigh|nine|(?:tw|sev)en|(?:thi|fo)r)ty['-]?(?:f(?:ive|our)|s(?:even|ix)|t(?:hree|wo)|(?:ni|o)ne|eight)\spercentage points",row['sentences'])
    for rr in r1,r2,r3,r4,r5:
        percentage_num.append(rr)
    for nn in n1,n2,n3,n4,n5,n6,n7,n8:
        percentage_word.append(nn)
    
percentage_num=    [x for x in percentage_num if x != []]
percentage_word=[x for x in percentage_word if x != []]
  
#need to flattening 
flattened_num=[]
flattened_word=[]
for x in percentage_num:
    for y in x:
        flattened_num.append(y)
for x in percentage_word:
    for y in x:
        flattened_word.append(y)
outperct=flattened_num+flattened_word
output_perct=pd.DataFrame(outperct)
output_perct.to_csv("output_perctage.csv")

#--------------CEO &Company-----------

ceo_pos_df=pd.read_csv('C:\\Users\\Amy\\Downloads\\all\\ceo.csv',encoding='ISO-8859-1',header=None)
ceo_pos_df.fillna('', inplace=True)
ceo_pos_df['fullname']=ceo_pos_df[0]+" "+ceo_pos_df[1]
company_df=pd.read_csv('C:\\Users\\Amy\\Downloads\\all\\companies.csv',encoding='ISO-8859-1',header=None)

df_clean_copy['pos']=pos_tag_sents(df_clean_copy['sentences'].apply(word_tokenize).tolist())
df_clean_copy.to_csv('savepos.csv')

# =============IF KERNEL DIES================================================================
#df_clean_copy=pd.read_csv('C:\\Users\\Amy\\.spyder-py3\\savepos.csv')
#df_clean_copy.drop(df_clean_copy.columns[[0]],axis=1,inplace=True)
# =============================================================================

#------------------CEO------------
df_clean_ceo=df_clean_copy.copy()
X_train, X_test= train_test_split(df_clean_ceo, test_size=0.4)
print (X_train.shape, X_test.shape)

ceo_pos_list=ceo_pos_df['fullname'].unique().tolist()

# clean stockword"


stop_wordsceo_raw = stopwords.words('english')
commonstopword=['San Francisco','Los Angeles','New York','United States','President','Members','On','Twitter','In','Police','The','London','City','China','Japan','Europe','European','Chinese','Governor','Chicago','Hong Kong','White House','Jackson Hole']
ceostopwordceo=['Goldman Sachs','Morgan Stanley','Wells Fargo','Google Finance','Thomson Reuters','Business Insider','Yahoo Finance','Time Warner','Wall Street','Deutsche Bank','Credit Suisse','Elliott Management','Federal Open','Merrill Lynch','Capital Management']
ceostopword=['Justice','Department','Federal Reserve','Central Bank','Communist','Vice Chairman','Vice President','University','Bank','Investment','Company','School']
stop_wordsceo_raw.extend(commonstopword)
stop_wordsceo_raw.extend(ceostopwordceo)
stop_wordsceo_raw.extend(ceostopword)
print(stop_wordsceo_raw)

def ceo_in_or_not(sentence_string):
    if any(s in sentence_string for s in ceo_pos_list):
        contain_ceo=1
    else:
        contain_ceo=0
    return contain_ceo


def get_rid_of_quotes(string_pos):
    x=ast.literal_eval(string_pos)
    return x

X_train['ceo_in']= X_train['sentences'].apply(ceo_in_or_not)
X_train['pos_clean']=X_train['pos'].apply(get_rid_of_quotes)
del X_train['pos']
X_traincopy=X_train.copy()

X_train_raw_pos=X_train.loc[X_train['ceo_in']==1]

#in contained_name, find true sample

def get_feature_ready_step1(dataframe):
    
    train_pos_feature=pd.DataFrame(columns=['word string','contain_ceo_word','sentence_index','words_index','y'])
    index1=0
    for i in range(len(dataframe)):
        if (i%10000==0):
            print(i)
        nounpattern=re.findall(r"[A-Z][a-z]+\s[A-Z][a-z]+",dataframe.iloc[i][0])
        if len(nounpattern)>0: # has two words capitalized,not empty string
            for eachstring in nounpattern:
                if eachstring in stop_wordsceo_raw:
                    continue
                contain_ceo_word=dataframe.iloc[i][0].find("CEO")
                if contain_ceo_word==-1:
                    boolvalue=0
                else:
                    boolvalue=1
                firstword=eachstring.split()[0]
                secondword=eachstring.split()[0]
                if firstword in stop_wordsceo_raw:
                    continue
                if secondword in stop_wordsceo_raw:
                    continue
                tokens_text=word_tokenize(dataframe.iloc[i][0])
                if firstword not in tokens_text:
                    continue
                first_name_index=tokens_text.index(firstword)
                if eachstring in ceo_pos_list:
                    train_pos_feature.loc[index1]=[eachstring,boolvalue,i,first_name_index,1]
                else:
                    train_pos_feature.loc[index1]=[eachstring,boolvalue,i,first_name_index,0]
                index1=index1+1
    return train_pos_feature

def test_filter_stopwords(row):
    word=row['word string']
    if word in stop_wordsceo_raw:
        filterval=1
    else:
        word_list=row['word string'].split()
        for item in word_list:
            if item in stop_wordsceo_raw:
                filterval=1
            else:
                filterval=0
    return filterval
                
    
    
def if_two_words_are_non(row):
    word_list=row['word string'].split()
    indexnum=row['sentence_index']
    dfdict=dict(X_train_raw_pos.iloc[indexnum]['pos_clean'])
    boolvalue =all(dfdict.get(item)=='NNP' for item in word_list)
    return boolvalue

def if_sent_contains_chief_executive_officer(row):
    indexnum=row['sentence_index']
    sentence=X_train_raw_pos.iloc[indexnum]['sentences']
    words=word_tokenize(sentence)
    lowercasewords=map(lambda x:x.lower(),words)
    if "chief" in lowercasewords:
        contain_chief=1
    else:
        contain_chief=0
    if "executive" in lowercasewords:
        contain_exec=1
    else:
        contain_exec=0
    if "officer" in lowercasewords:
        contain_officer=1
    else:
        contain_officer=0
    return contain_chief,contain_exec,contain_officer

def return_first_len(row):
    firstword=row['word string'].split()[0]
    first_name_length=len(firstword)  
    return first_name_length      

#train_pos_feature['contain filter']=train_pos_feature.apply(filter_stopwords,axis=1)
#train_pos_feature_afterfilter=train_pos_feature.loc[train_pos_feature['contain filter']==0]
#del train_pos_feature_afterfilter['contain filter']
train_pos_feature=get_feature_ready_step1(X_train_raw_pos)
train_pos_feature['bothnnp']=train_pos_feature.apply(if_two_words_are_non,axis=1)     
train_pos_feature['first_name_length']=train_pos_feature.apply(return_first_len,axis=1)   
testa=train_pos_feature.apply(if_sent_contains_chief_executive_officer,axis=1)
testdf=testa.apply(pd.Series)   
train_pos_feature['contain_chief']=testdf.iloc[:,0]
train_pos_feature['contain_exec']=testdf.iloc[:,1]
train_pos_feature['contain_officer']=testdf.iloc[:,2]

len(train_pos_feature)        


#need 1-1 ratio

resultlen=train_pos_feature['y'].value_counts().to_frame()
print (resultlen)
posnum=int(resultlen.iloc[1])
negnum=int(resultlen.iloc[0])
cleaned_pos_train_sample=train_pos_feature.loc[train_pos_feature['y']==1]
cleaned_neg_train_sample=train_pos_feature.loc[train_pos_feature['y']==0]
if posnum<negnum:
    neg_new_size=posnum
    negative_updated_sample=cleaned_neg_train_sample.iloc[:neg_new_size]

train_shrink=pd.concat([cleaned_pos_train_sample,negative_updated_sample])
train_shrink_copy=train_shrink.copy()
train_shrink_copy.head(10)
train_shrink.to_csv('train_ceo_backup.csv')



# =============================================================================
# #get test data
# test_pos_feature=pd.DataFrame(columns=['word string','contain_ceo_word','index','y'])
# X_test['pos_clean']=X_test['pos'].apply(get_rid_of_quotes)
# del X_test['pos']
# 
# for i in range(len(X_test)):
#     if (i%10000==0):
#         print(i)
#     nounpattern=re.findall(r"[A-Z][a-z]+\s[A-Z][a-z]+",X_test.iloc[i][0])
#     if len(nounpattern)>0: # has two words capitalized,not empty string
#         for eachstring in nounpattern:
#             contain_ceo_word=X_test.iloc[i][0].find("CEO")
#             if contain_ceo_word==-1:
#                 boolvalue=0
#             else:
#                 boolvalue=1
#             if eachstring in ceo_pos_list:
#                 test_pos_feature.loc[index1]=[eachstring,boolvalue,i,1]
#             else:
#                 test_pos_feature.loc[index1]=[eachstring,boolvalue,i,0]
#             index1=index1+1
# test_pos_feature['bothnnp']=test_pos_feature.apply(if_two_words_are_non,axis=1)        
# test_pos_feature_copy=test_pos_feature.copy()                   
# =============================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X_ceo=train_shrink_copy[['contain_ceo_word','bothnnp','words_index','first_name_length','contain_chief','contain_exec','contain_officer']]

X_ceo['contain_ceo_word']=X_ceo['contain_ceo_word'].astype('int')
Y_ceo=train_shrink_copy[['y']]
Y_ceo['y']=Y_ceo['y'].astype('bool')
scalar = StandardScaler()
scalar.fit(X_ceo)
scaledx_ceo=scalar.transform(X_ceo)


test_df=train_pos_feature.copy()
test_df['contain_filter']=test_df.apply(test_filter_stopwords,axis=1)
test_df=test_df.loc[test_df['contain_filter']==0]
del test_df['contain_filter']
X_test_ceo=test_df[['contain_ceo_word','bothnnp','words_index','first_name_length','contain_chief','contain_exec','contain_officer']]
X_test_ceo['contain_ceo_word']=X_test_ceo['contain_ceo_word'].astype('int')
Y_test_ceo=test_df[['y']]
Y_test_ceo['y']=Y_test_ceo['y'].astype('bool')
scalar2 = StandardScaler()
scalar2.fit(X_test_ceo)
scaledx_test_ceo=scalar.transform(X_test_ceo)


#----logistic regression
model_ceo=LogisticRegression(random_state=0,solver='lbfgs',multi_class='multinomial').fit(scaledx_ceo,Y_ceo)
y_pred=model_ceo.predict(scaledx_test_ceo) #testdata
b=accuracy_score(Y_test_ceo,y_pred)
a = precision_recall_fscore_support(Y_test_ceo, y_pred, average='binary')


#----randomeforest
from sklearn.ensemble import RandomForestClassifier
rfceo=RandomForestClassifier(n_estimators=500)
rf_fit=rfceo.fit(scaledx_ceo,Y_ceo)
rf_pred_test=rf_fit.predict(scaledx_test_ceo)
rf_score=rf_fit.score(scaledx_test_ceo,Y_test_ceo)
rf_metrics=precision_recall_fscore_support(Y_test_ceo,rf_pred_test,average='macro')

print ("Logistic Regression for ceos: accuracy score is {}".format(b))
print("Logistic Regression for ceos: precision is {}, recall is {}, and f1 score is {}.".format(a[0],a[1],a[2]))
print ("Random Forest for ceos: accuracy score is {}".format(rf_score))
print("Random Forest  for ceos: precision is {}, recall is {}, and f1 score is {}.".format(rf_metrics[0],rf_metrics[1],rf_metrics[2]))

resulttest=pd.concat([pd.DataFrame(test_df[['word string','y']]),pd.DataFrame(rf_pred_test)],axis=1)
fakepostive =resulttest.loc[(resulttest['y']==0)&resulttest[0]==True]
fakepositive=fakepostive.groupby('word string').count()
fakepositive=fakepositive.sort_index(ascending=False)
fakepositive.to_csv('checkdictionary.csv')

# ================use to predict whole dataset=============================================================
wholedf=df_clean_ceo.copy()

wholedf['ceo_in']= wholedf['sentences'].apply(ceo_in_or_not)
wholedf['pos_clean']=wholedf['pos'].apply(get_rid_of_quotes)
del wholedf['pos']
wholedf_copy=wholedf.copy()

wholedf_test_feature=get_feature_ready_step1(wholedf_copy)

wholedf_test_feature['contain_filter']=wholedf_test_feature.apply(test_filter_stopwords,axis=1)
wholedf_test_feature=wholedf_test_feature.loc[wholedf_test_feature['contain_filter']==0]
del wholedf_test_feature['contain_filter']


    
def if_two_words_are_non_whole(row):
    word_list=row['word string'].split()
    indexnum=row['sentence_index']
    dfdict=dict(wholedf.iloc[indexnum]['pos_clean'])
    boolvalue =all(dfdict.get(item)=='NNP' for item in word_list)
    return boolvalue

def if_sent_contains_chief_executive_officer_whole(row):
    indexnum=row['sentence_index']
    sentence=wholedf.iloc[indexnum]['sentences']
    words=word_tokenize(sentence)
    lowercasewords=map(lambda x:x.lower(),words)
    if "chief" in lowercasewords:
        contain_chief=1
    else:
        contain_chief=0
    if "executive" in lowercasewords:
        contain_exec=1
    else:
        contain_exec=0
    if "officer" in lowercasewords:
        contain_officer=1
    else:
        contain_officer=0
    return contain_chief,contain_exec,contain_officer

def return_first_len_whole(row):
    firstword=row['word string'].split()[0]
    first_name_length=len(firstword)  
    return first_name_length      



wholedf_test_feature['bothnnp']=wholedf_test_feature.apply(if_two_words_are_non_whole,axis=1)     
wholedf_test_feature['first_name_length']=wholedf_test_feature.apply(return_first_len_whole,axis=1)   
wholetesta=wholedf_test_feature.apply(if_sent_contains_chief_executive_officer_whole,axis=1)
wholetestadf=testa.apply(pd.Series)   
wholedf_test_feature['contain_chief']=wholetestadf.iloc[:,0]
wholedf_test_feature['contain_exec']=wholetestadf.iloc[:,1]
wholedf_test_feature['contain_officer']=wholetestadf.iloc[:,2]

wholedf_test_feature_copy=wholedf_test_feature.copy()
wholedf_test_feature_copy=wholedf_test_feature_copy.drop_duplicates(subset='word string', keep="last")

X_finaltest_ceo= wholedf_test_feature_copy[['contain_ceo_word','bothnnp','words_index','first_name_length','contain_chief','contain_exec','contain_officer']]
X_finaltest_ceo['contain_ceo_word']=X_finaltest_ceo['contain_ceo_word'].astype('int')
X_finaltest_ceo['words_index']=X_finaltest_ceo['words_index'].astype('int')
Y_finaltest_ceo=wholedf_test_feature_copy[['y']]
Y_finaltest_ceo['y']=Y_finaltest_ceo['y'].astype('bool')


X = X_finaltest_ceo.as_matrix().astype(np.float)
Y = Y_finaltest_ceo.as_matrix().astype(np.float)
X=np.nan_to_num(X)

print(X[:,0].shape)
index = 0
for i in X[:,0]:
    if not np.isnan(i):
        print(index, i)
    index +=1


scalar2 = StandardScaler()
scalar2.fit(X)
scaled_final_test_ceo=scalar.transform(X)
rf_pred_test_final=rf_fit.predict(scaled_final_test_ceo)
resulttest=pd.DataFrame()
wholedf_test_feature_copy.reset_index(inplace=True,drop=True) 

resultfinal=pd.concat([pd.DataFrame(wholedf_test_feature_copy[['word string']]),pd.DataFrame(rf_pred_test_final)],axis=1,ignore_index=True)
predicted_ceo=resultfinal[resultfinal[1]==True]
predicted_ceo=predicted_ceo.sort_index(ascending=True)
predicted_ceo.to_csv('output_ceo.csv')
#need drop duplicates

#--------------------------Company---------------

company_df.columns=['company names']
company_df_sample=company_df.drop_duplicates(subset='company names', keep="last")
company_df_sample['capitalized']=list(map(lambda x: x[0].isupper(),company_df_sample['company names']))
notcapitalizedf=company_df_sample.loc[company_df_sample['capitalized']==False]
print(notcapitalizedf)

companysamplelist=company_df_sample['company names'].tolist()
df_clean_company=df_clean_copy.copy()
X_train_company, X_test_company= train_test_split(df_clean_company, test_size=0.4)
print (X_train_company.shape, X_test_company.shape)

X_train_company['pos_clean']=X_train_company['pos'].apply(get_rid_of_quotes)
del X_train_company['pos']
X_train_company_copy=X_train_company.copy()


df1 = X_train_company_copy.iloc[:80000,:]
df2 = X_train_company_copy.iloc[80000:160000,:]
df3=X_train_company_copy.iloc[160000:240000,:]
df4=X_train_company_copy.iloc[240000:320000,:]
df5=X_train_company_copy.iloc[320000:,:]

def get_feature_step1(dataframe):
    train_pos_feature=pd.DataFrame(columns=['word string','sentence_index','y'])
    index1=0
    for i in range(len(dataframe)):
         if (i%10000==0):
             print(i)
         nounpattern=re.findall(r"[A-Z][\w-]+(?:(?:\s+[A-Z][\w-]*)+)",dataframe.iloc[i][0])
        
         if len(nounpattern)>0:
             for eachstring in nounpattern:
                 if eachstring in companysamplelist:
                     train_pos_feature.loc[index1]=[eachstring,i,1]
                 else:
                     train_pos_feature.loc[index1]=[eachstring,i,0]
         index1=index1+1
    return train_pos_feature


feature_1=get_feature_step1(df1)
feature_1=feature_1.drop_duplicates(subset='word string',keep="last")
feature_2=get_feature_step1(df2)
feature_2=feature_2.drop_duplicates(subset='word string',keep="last")
feature_3=get_feature_step1(df3)
feature_3=feature_3.drop_duplicates(subset='word string',keep="last")
feature_4=get_feature_step1(df4)
feature_4=feature_4.drop_duplicates(subset='word string',keep="last")
feature_5=get_feature_step1(df5)
feature_5=feature_5.drop_duplicates(subset='word string',keep="last")


def loop_through(featurematrix,dataframe):
    dataframetest=dataframe.copy()
    print(dataframetest.iloc[0][0])

    def if_sent_contains_company_keyword(row):
        indexnum=row['sentence_index']
    
        #print(len(dataframetest))
        #print (indexnum)
        sentence=dataframetest.iloc[indexnum]['sentences']
        sentence_words=word_tokenize(sentence)
        keywords = {'Inc', 'Group', 'Ltd', 'Co', 'Corp', 'Corporation','Management','Company'}
        #lowercasewords=map(lambda x:x.lower(),words)
        if "company" in sentence_words:
            contain_company=1
        else:
            contain_company=0
        if any(s in sentence_words for s in keywords):
            contain_keyword=1
        else:
            contain_keyword=0
        first_word=row['word string'].split()[0]
        if first_word not in sentence_words:
            first_name_index=-1 # not in the list, should not be included when predicting
        else:
            first_name_index=sentence_words.index(first_word) 
        return contain_company,contain_keyword,first_name_index
    
    def length_of_company_words(row): #length of company name(howmanyletters), how many words in it
        word_list=row['word string'].split()
        num_of_words=len(word_list)
        num_of_letters=sum(len(x) for x in word_list)
        return num_of_words,num_of_letters
        
    teasta=featurematrix.apply(if_sent_contains_company_keyword,axis=1)
    testdf=teasta.apply(pd.Series)    
    featurematrix['contain_company']=testdf.iloc[:,0]
    featurematrix['contain_keyword']=testdf.iloc[:,1]
    featurematrix['first_word_index']=testdf.iloc[:,2]
    teastb=featurematrix.apply(length_of_company_words,axis=1)
    testdf2=teastb.apply(pd.Series)    
    featurematrix['number_of_words']=testdf2.iloc[:,0]
    featurematrix['num_of_letters']=testdf2.iloc[:,1]
    return featurematrix

feature_1=loop_through(feature_1,df1)
feature_2=loop_through(feature_2,df2)
feature_3=loop_through(feature_3,df3)
feature_4=loop_through(feature_4,df4)
feature_5=loop_through(feature_5,df5)
#balance train pos negative sample

whole_feature_matrix=pd.concat([feature_1,feature_2,feature_3,feature_4,feature_5])
whole_feature_matrix=whole_feature_matrix[whole_feature_matrix['first_word_index']!=-1]
whole_feature_matrix=whole_feature_matrix.reset_index()

stopwordlistcomp={'Today','Judge','January','February','March','April','May','June','July','August','September','October','November',\
                  'December','Not','No','My','Ms','Mrs','Mr','Maybe','Meanwhile','FREE','Japanese','Chinese','In','If','Is','IS','From',\
                  'Fed','Governor','Government','Federal','Great','Grand','Has','Here','Officer','Can','But','At','As','Chairman','Chief'}

whole_feature_matrix_copy=whole_feature_matrix.copy()
filter_whole_feature=pd.DataFrame(columns=whole_feature_matrix_copy.columns)


index2=0
for index,row in whole_feature_matrix_copy.iterrows():
    if (index%10000==0):
        print (index)
    word_list=row['word string'].split()
    containwordlist=any(word in word_list for word in stopwordlistcomp)
    if containwordlist==False:
        if row['word string'] not in ceo_pos_list:
            filter_whole_feature.loc[index2]=whole_feature_matrix_copy.loc[index]
            index2=index2+1


whole_feature_filter_copy=filter_whole_feature.copy()
whole_feature_filter_copy.head(10)

resultlen_cop=whole_feature_filter_copy['y'].value_counts().to_frame()
print (resultlen_cop)
posnum_cop=int(resultlen_cop.iloc[1])
negnum_cop=int(resultlen_cop.iloc[0])
cleaned_pos_train_sample_cop=whole_feature_filter_copy.loc[whole_feature_matrix['y']==1]
cleaned_neg_train_sample_cop=whole_feature_filter_copy.loc[whole_feature_matrix['y']==0]
if posnum_cop<negnum_cop:
    neg_new_size_2=posnum_cop
    negative_updated_sample_cop=cleaned_neg_train_sample_cop.iloc[:neg_new_size_2]

train_shrink_copmany=pd.concat([cleaned_pos_train_sample_cop,negative_updated_sample_cop])
train_shrink_company_copy=train_shrink_copmany.copy()


X_comp=train_shrink_company_copy[['contain_company','contain_keyword','first_word_index','number_of_words','num_of_letters']]

X_comp['first_word_index']=X_comp['first_word_index'].astype('int')
Y_comp=train_shrink_company_copy[['y']]
Y_comp['y']=Y_comp['y'].astype('bool')
scalarcopm = StandardScaler()
scalarcopm.fit(X_comp)
scaledx_comp=scalarcopm.transform(X_comp)

#-------test whole train matrix
X_test_comp=whole_feature_filter_copy[['contain_company','contain_keyword','first_word_index','number_of_words','num_of_letters']]
X_test_comp['first_word_index']=X_test_comp['first_word_index'].astype('int')
Y_test_comp=whole_feature_filter_copy[['y']]
Y_test_comp['y']=Y_test_comp['y'].astype('bool')
scalarcomp_test = StandardScaler()
scalarcomp_test.fit(X_test_comp)
scaledx_comptest=scalarcomp_test.transform(X_test_comp)

from sklearn.ensemble import RandomForestClassifier
rfcomp=RandomForestClassifier(n_estimators=500)
rf_fit_comp=rfcomp.fit(scaledx_comp,Y_comp)


rf_pred_test_comp=rf_fit_comp.predict(scaledx_comptest)
rf_score=rf_fit_comp.score(scaledx_comptest,Y_test_comp)
rf_metrics=precision_recall_fscore_support(Y_test_comp,rf_pred_test_comp,average='macro')

print ("Random Forest for companies: accuracy score is {}".format(rf_score))
print("Random Forest  for companies: precision is {}, recall is {}, and f1 score is {}.".format(rf_metrics[0],rf_metrics[1],rf_metrics[2]))


resulttest2=pd.concat([pd.DataFrame(whole_feature_filter_copy[['word string']]),pd.DataFrame(rf_pred_test_comp)],axis=1)

#---------entire dataset
X_test_company_copy=X_test_company.copy()
df6 = X_test_company_copy.iloc[:80000,:]
df7 = X_test_company_copy.iloc[80000:160000,:]
df8=X_test_company_copy.iloc[160000:,:]

feature_6=get_feature_step1(df6)
feature_6=feature_6.drop_duplicates(subset='word string',keep="last")
feature_7=get_feature_step1(df7)
feature_7=feature_7.drop_duplicates(subset='word string',keep="last")
feature_8=get_feature_step1(df8)
feature_8=feature_8.drop_duplicates(subset='word string',keep="last")


feature_6=loop_through(feature_6,df6)
feature_7=loop_through(feature_7,df7)
feature_8=loop_through(feature_8,df8)

whole_feature_matrix_test=pd.concat([feature_6,feature_7,feature_8])
whole_feature_matrix_test=whole_feature_matrix_test[whole_feature_matrix_test['first_word_index']!=-1]
whole_feature_matrix_test=whole_feature_matrix_test.reset_index()

whole_feature_matrix_test_copy=whole_feature_matrix_test.copy()
filter_whole_feature_test=pd.DataFrame(columns=whole_feature_matrix_test_copy.columns)

index3=0
for index,row in whole_feature_matrix_test_copy.iterrows():
    if (index%10000==0):
        print (index)
    word_list=row['word string'].split()
    containwordlist=any(word in word_list for word in stopwordlistcomp)
    if containwordlist==False:
        if row['word string'] not in ceo_pos_list:
            filter_whole_feature_test.loc[index3]=whole_feature_matrix_test_copy.loc[index]
            index3=index3+1

X_company_test_whole=filter_whole_feature_test.copy()

X_test_complete_comp=X_company_test_whole[['contain_company','contain_keyword','first_word_index','number_of_words','num_of_letters']]
X_test_complete_comp['first_word_index']=X_test_comp['first_word_index'].astype('int')

scalarcomp_testcomp = StandardScaler()
scalarcomp_testcomp.fit(X_test_complete_comp)
scaledx_completetestcomp=scalarcomp_testcomp.transform(X_test_complete_comp)

#use previously fitted model
rf_pred_test_complete_comp=rf_fit_comp.predict(scaledx_completetestcomp)

resulttestfinal=pd.concat([pd.DataFrame(filter_whole_feature_test[['word string']]),pd.DataFrame(rf_pred_test_complete_comp)],axis=1,ignore_index=True)
resulttestfinal.columns=['word string',0]
resultcompfinaltotal=pd.concat([resulttest2,resulttestfinal])
predicted_comp=resultcompfinaltotal[resultcompfinaltotal[0]==True]
predicted_comp=predicted_comp.drop_duplicates(subset='word string',keep="last")
predicted_comp=predicted_comp.sort_index(ascending=True)
predicted_comp.to_csv('output_company.csv')

