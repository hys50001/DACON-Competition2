# DACON-Competition2

![image](https://user-images.githubusercontent.com/75110162/103340444-6b0ea000-4ac7-11eb-9818-0ebbc5bcf35e.png)

두번째로 참가한 DACON Competition, 대회 참가하면서 많이 배우고 좋은 결과까지 이어져 뜻깉은 경험

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
