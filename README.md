# DACON-Competition2

![image](https://user-images.githubusercontent.com/75110162/103340444-6b0ea000-4ac7-11eb-9818-0ebbc5bcf35e.png)

두번째로 참가한 DACON Competition, 대회 참가하면서 많이 배우고 좋은 결과까지 이어져 뜻깊은 경험

![image](https://user-images.githubusercontent.com/75110162/103340660-fb4ce500-4ac7-11eb-80d9-666b9b1eea91.png)

- 소설 작가의 문체가 주어지고 5명의 작가 중 어느 작가의 문체인지 분류 하는 __Multi-Class Classification__ 과제 

--------

### 모델1. LSTM

#### STEP1-1: 전처리: 문장부호 제거 
``` python
def alpha_num(text):
    return re.sub(r'[^A-Za-z0-9 ]', '', text)

train['text']=train['text'].apply(alpha_num)
```
#### STEP1-2: 불용어 제거 : nltk library 의 stopwords 이용 
``` python
nltk.download('stopwords')
eng_stopwords = set(stopwords.words("english"))
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stopwords:
            final_text.append(i.strip())
    return " ".join(final_text)
```

#### STEP2: 모델링: LSTM : Keras library 이용 
``` python
model = Sequential()
model.add(Embedding(len(word_index)+1, output_dim=embed_size, input_length=MAX_LEN))
model.add(LSTM(units=128 , return_sequences = True , recurrent_dropout = 0.25 , dropout = 0.25))
model.add(LSTM(units=64 , recurrent_dropout = 0.1 , dropout = 0.1))
model.add(Dense(units = 32 , activation = 'relu'))
model.add(Dense(5,activation='softmax'))
model.compile(optimizer=keras.optimizers.Adam(lr = 0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
#### STEP3: 결과
![image](https://user-images.githubusercontent.com/75110162/103350053-10367200-4ae2-11eb-9f12-026e6ba3438e.png)

베이스라인 보다는 loss가 낮았지만, 모델에 layer를 추가하여 복잡하게 만들수록 과적합이 일어나 loss 점점 높아짐..

#### 원인분석
- 문장 부호를 제거하는 전처리를 진행하면 작가 고유의 특성이 사라지게됨 
- stopwords 역시 마찬가지, stopwords 에 등장하는 단어를 자주 쓰는 것 역시 작가의 고유 특성이 될 수 있음 
- 한가지의 모델로 loss를 낮추는 데에는 한계가 있다고 생각, Ensemble 기법 필요하다. 

---------------

### 모델2. XGBoost
Concept:  **Meta Feature**와 **Text Based Feature**로 구분지어 Feature Engineering 후 Stacking Ensemble 기법 활용

#### STEP1: META Feature
사용한 대표적 Meta Feature(중요도가 높았던 Feature)는 다음과 같다. 
- 단어 갯수
- 평균 단어 길이
- 음절 갯수
- 쉼표 사이의 음절 갯수
- 명사 비율
- 형용사 비율
- 문장 난이도 (flesch_reading_ease library 활용)
- 문장 감성 분석( nltk SentimentIntensityAnalyzer 활용)
- POS tagging 을 통해 문장에서 등장하는 고유 명사(사람 이름 등) 중복도 

``` python
train['num_words']=train['text'].apply(lambda x:len(get_words(x)))
train['mean_word_len']=train['text'].apply(lambda x:np.mean([len(w) for w in str(x).split()]))
train["num_unique_words"] = train["text"].apply(lambda x: len(set(str(x).split())))
train["num_chars"] = train["text"].apply(lambda x: len(str(x)))
train["num_stopwords"] = train["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
train["num_punctuations"] =train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
train["num_words_upper"] = train["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))/train["num_words"]
train["num_words_title"] = train["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))/train["num_words"]
train["chars_between_comma"] = train["text"].apply(lambda x: np.mean([len(chunk) for chunk in str(x).split(",")]))/train["num_chars"]
train["symbols_unknowns"]=train["text"].apply(lambda x: np.sum([not w in symbols_knowns for w in str(x)]))/train["num_chars"]
train['noun'] = train["text"].apply(lambda x: fraction_noun(x))
train['adj'] = train["text"].apply(lambda x: fraction_adj(x))
train['verbs'] = train["text"].apply(lambda x: fraction_verbs(x))
train["sentiment"]=train["text"].apply(sentiment_nltk)
train['single_frac'] = train['text'].apply(lambda x: count_tokens(x, ['is', 'was', 'has', 'he', 'she', 'it', 'her', 'his']))/train["num_words"]
train['plural_frac'] = train['text'].apply(lambda x: count_tokens(x, ['are', 'were', 'have', 'we', 'they']))/train["num_words"]
train['first_word_len']=train['text'].apply(first_word_len)/train["num_chars"]
train['last_word_len']=train['text'].apply(last_word_len)/train["num_chars"]
train["first_word_id"] = train['text'].apply(lambda x: symbol_id(list(x.strip())[0]))
train["last_word_id"] = train['text'].apply(lambda x: symbol_id(list(x.strip())[-1]))
train['ease']=train['text'].apply(flesch_reading_ease)
```

#### STEP2: Text Based Feature
여러가지 Machine Learning Algorithm들의 결과를 Stacking 하여 loss를 줄여나갔다.
사용한 기법은 다음과 같다.

- LogisticRegression
- SGDClassifier
- RandomForestClassifier
- MLPClassifier
- DecisionTreeClassifier

``` python
tfidf_vec = TfidfVectorizer(tokenizer=word_tokenize, stop_words=stopwords.words('english'), ngram_range=(1, 3), min_df=50)
train_tfidf = tfidf_vec.fit_transform(train['text'].values.tolist())
test_tfidf = tfidf_vec.transform(test['text'].values.tolist())
train_y = train['author']

def runLR(train_X,train_y,test_X,test_y,test_X2):
    model=LogisticRegression()
    model.fit(train_X,train_y)
    pred_test_y=model.predict_proba(test_X)
    pred_test_y2=model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model


cv_scores=[]
cols_to_drop=['text','index']
train_X = train.drop(cols_to_drop+['author'], axis=1)
train_y=train['author']
test_X = test.drop(cols_to_drop, axis=1)
pred_train=np.zeros([train.shape[0],5])
pred_full_test = 0

cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)

for dev_index, val_index in cv.split(train_X,train_y):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runLR(dev_X, dev_y, val_X, val_y,test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5.

train["tfidf_LR_0"] = pred_train[:,0]
train["tfidf_LR_1"] = pred_train[:,1]
train["tfidf_LR_2"] = pred_train[:,2]
train["tfidf_LR_3"] = pred_train[:,3]
train["tfidf_LR_4"] = pred_train[:,4]
test["tfidf_LR_0"] = pred_full_test[:,0]
test["tfidf_LR_1"] = pred_full_test[:,1]
test["tfidf_LR_2"] = pred_full_test[:,2]
test["tfidf_LR_3"] = pred_full_test[:,3]
test["tfidf_LR_4"] = pred_full_test[:,4]
```
**TFIDF vectorizer + Logistic Regression 을 활용한 Feature Stacking**
