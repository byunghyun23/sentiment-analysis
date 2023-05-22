from konlpy.tag import Okt, Kkma
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Bidirectional, Dropout, Multiply, Add, Concatenate, Attention, \
    MultiHeadAttention, Reshape, GlobalAveragePooling1D
from gensim.models.fasttext import FastText
from hanspell import spell_checker
import datetime
import time
import gensim
import pickle
import numpy as np
import urllib.request
import tensorflow as tf
import pandas as pd


def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round((elapsed)))
    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))

# option
data_write_option = True
score_option = True

# 0. set GPU
print('========== Set GPU ==========')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print('logical_gpus', logical_gpus)
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# 1. download data
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")


# 2. load data
print('========== Load data ==========')
train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')
# train_data = train_data[:1000]
# test_data = test_data[:1000]
print('Length of Train:', len(train_data))
print('Length of Test:', len(test_data))
train_data['document'].nunique(), train_data['label'].nunique()
train_data.drop_duplicates(subset=['document'], inplace=True)
print('Length of Total:', len(train_data))


# 3. pre-processing
print('========== Pre-processing ==========')
# train_data = train_data.dropna(how='any') # Remove rows with Null
# print(train_data.isnull().values.any()) # Check Null

# train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
# train_data['document'] = train_data['document'].str.replace('[-=+#/\:^$@*\"※&%ㆍ』\\‘|\(\)\[\]\<\>`\'…》]', '')
train_data['document'] = train_data['document'].str.replace('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '')
train_data['document'] = train_data['document'].str.replace('^ +', "")
train_data['document'].replace('', np.nan, inplace=True)
train_data = train_data.dropna(how='any')
print('Length of Train after pre-processing:', len(train_data))


test_data.drop_duplicates(subset = ['document'], inplace=True)
# test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
# test_data['document'] = test_data['document'].str.replace('[-=+#/\:^$@*\"※&%ㆍ』\\‘|\(\)\[\]\<\>`\'…》]', '')
test_data['document'] = test_data['document'].str.replace('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '')
test_data['document'] = test_data['document'].str.replace('^ +', "")
test_data['document'].replace('', np.nan, inplace=True)
test_data = test_data.dropna(how='any')
print('Length of Test after pre-processing:', len(test_data))


# 4. build Sentiment Dictionary
print('========== Build Sentiment Dictionary ==========')
my_dic = {}
my_dic_adj = {}
my_dic_adv = {}
sigmoid_my_dic_adj = {}
sigmoid_my_dic_adv = {}

encoder = LabelEncoder()
encoder.fit(train_data.label.tolist())

df_train_target_label = encoder.transform(train_data.label.tolist())

if score_option is True:
    t0 = time.time()
    print('Start build..')
    okt = Okt()
    # okt = Kkma()
    train_pos_list = []
    train_max = len(train_data.document)
    pos_tag_cnt = 0

    spell_checker_list = []

    for text, y in zip(train_data.document, df_train_target_label):
        # okt = Kkma()

        pos_tag_cnt += 1
        if pos_tag_cnt % 1000 == 0:
            print('processing status:', pos_tag_cnt, '/', train_max)
        sign = 1
        if y == 0:
            sign = -1

        try:
            text = spell_checker.check(text).checked
        except Exception as check_exception:
            pass

        text = okt.normalize(text)
        spell_checker_list.append(text)

        pos_tag_data_list = []
        try:
            pos_tag_data_list = okt.pos(text, stem=True)
        except Exception as okt_exception:
            print('okt error', text)

        # NNG 일반 명사
        # NNP 고유 명사
        # VV 동사
        # VA 형용사
        # MAG 일반 부사
        token_cnt = 0
        for pos_tag_data in pos_tag_data_list:
            # if len(pos_tag_data[0]) > 1 and (pos_tag_data[1] == 'Adjective'):
            # if len(pos_tag_data[0]) > 1 and (pos_tag_data[1] == 'Adverb'):
            # if len(pos_tag_data[0]) > 1 and (pos_tag_data[1] == 'Adjective' or pos_tag_data[1] == 'Adverb'):
            if len(pos_tag_data[0]) > 1 and (pos_tag_data[1] == 'Verb' or pos_tag_data[1] == 'Adjective' or pos_tag_data[1] == 'Adverb' or pos_tag_data[1] == 'Noun'):
            # if len(pos_tag_data[0]) > 1 and (
            #         pos_tag_data[1] == 'NNG' or pos_tag_data[1] == 'NNP' or pos_tag_data[1] == 'VV' or
            #         pos_tag_data[1] == 'VA' or pos_tag_data[1] == 'MAG'):
                token_cnt += 1
        # print('token_cnt', token_cnt)

        train_pos = []
        for pos_tag_data in pos_tag_data_list:
            # if len(pos_tag_data[0]) > 1 and (pos_tag_data[1] == 'Adjective'):
            # if len(pos_tag_data[0]) > 1 and (pos_tag_data[1] == 'Adverb'):
            # if len(pos_tag_data[0]) > 1 and (pos_tag_data[1] == 'Adjective' or pos_tag_data[1] == 'Adverb'):
            if len(pos_tag_data[0]) > 1 and (pos_tag_data[1] == 'Verb' or pos_tag_data[1] == 'Adjective' or pos_tag_data[1] == 'Adverb'or pos_tag_data[1] == 'Noun'):
            # if len(pos_tag_data[0]) > 1 and (
            #         pos_tag_data[1] == 'NNG' or pos_tag_data[1] == 'NNP' or pos_tag_data[1] == 'VV' or
            #         pos_tag_data[1] == 'VA' or pos_tag_data[1] == 'MAG'):
                train_pos.append(pos_tag_data[0])     # 품사 태깅한 단어만 추출

                if pos_tag_data[0] in my_dic:
                    my_dic[pos_tag_data[0]] = my_dic.get(pos_tag_data[0]) + ((1 / token_cnt) * sign)
                else:
                    my_dic[pos_tag_data[0]] = ((1 / token_cnt) * sign)

                # if pos_tag_data[1] == 'Adjective':
                #     if pos_tag_data[0] in my_dic_adj:
                #         my_dic_adj[pos_tag_data[0]] = my_dic_adj.get(pos_tag_data[0]) + ((1 / token_cnt) * sign)
                #     else:
                #         my_dic_adj[pos_tag_data[0]] = ((1 / token_cnt) * sign)
                # elif pos_tag_data[1] == 'Adverb':
                #     if pos_tag_data[0] in my_dic_adv:
                #         my_dic_adv[pos_tag_data[0]] = my_dic_adv.get(pos_tag_data[0]) + ((1 / token_cnt) * sign)
                #     else:
                #         my_dic_adv[pos_tag_data[0]] = ((1 / token_cnt) * sign)

        line_pos = ' '.join(train_pos)
        if '' != line_pos:
            train_pos_list.append(' '.join(train_pos))

    elapsed = format_time(time.time() - t0)
    print('End time of build:', elapsed)

    df_train_pos = pd.DataFrame({'document': train_pos_list})
    print('df_train.text[:5]')
    print(train_data.document[:5])
    print('df_train_pos.text[:5]')
    print(df_train_pos.document[:5])

    # save dic pickle
    with open('my_dic' + '_' + '.pickle', 'wb') as fw:
        pickle.dump(my_dic, fw)
    # save dic to txt
    with open('my_dic' + '_' + '.txt', 'w', encoding='UTF-8') as tfw:
        for code, name in my_dic.items():
            tfw.write(f'{code} : {name}\n')

    # save pandas
    df_train_pos.to_csv('df_train_pos' + '_' + '.csv', encoding="utf-8-sig")
    df_train_pos.to_csv('df_train_pos' + '_' + '.txt', encoding="utf-8-sig")

    spell_checker = pd.DataFrame({'document': spell_checker_list})
    spell_checker.to_csv('spell_checker.csv', encoding="utf-8-sig")
    spell_checker.to_csv('spell_checker.txt', encoding="utf-8-sig")
else:
    # load dic
    with open('my_dic' + '_' + '.pickle', 'rb') as fr:
        my_dic = pickle.load(fr)

    # load pandas
    df_train_pos = pd.read_csv('df_train_pos' + '_' + '.csv')


# 5. generation verctors based on Sentiment Dictionary
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print('========== Generation verctors based on Sentiment Dictionary ==========')
# sigmoid_my_dic
tokenized_key_list = []
tokenized_score_list = []

for key, value in my_dic.items():
    tokenized_key_list.append(key)
    tokenized_score_list.append([my_dic.get(key)])

tokenized_score_matrix = np.array(tokenized_score_list)
tokenized_score_matrix = sigmoid(tokenized_score_matrix)

sigmoid_my_dic = {}

for key, value in zip(tokenized_key_list, tokenized_score_matrix):
    sigmoid_my_dic[key] = value[0]

f = open('score.txt', 'w')
csv_f = open('score.csv', 'w')
# csv_wr = csv.writer(csv_f)

my_dic_sorted = sorted(my_dic.items(), key=lambda x: x[1], reverse=True)
for my_dic_data in my_dic_sorted:
    # save .txt
    data = my_dic_data[0] + ' ' + str(my_dic_data[1]) + '\n'
    f.write(data)

    # save .csv
    csv_f.write(my_dic_data[0] + ',' + str(my_dic_data[1]) + '\n')
f.close()
csv_f.close()

f = open('score_sigmoid.txt', 'w')
csv_f = open('score_sigmoid.csv', 'w')

sigmoid_my_dic_sorted = sorted(sigmoid_my_dic.items(), key=lambda x: x[1], reverse=True)
for sigmoid_my_dic_data in sigmoid_my_dic_sorted:
    # save .txt
    data = sigmoid_my_dic_data[0] + ' ' + str(sigmoid_my_dic_data[1]) + '\n'
    f.write(data)

    # save .csv
    csv_f.write(sigmoid_my_dic_data[0] + ',' + str(sigmoid_my_dic_data[1]) + '\n')
f.close()
csv_f.close()



# 6. generation train and test data
print('========== Generation train and test data ==========')
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
okt = Okt()
# okt = Kkma()
X_train = []
X_test = []
if data_write_option is True:
    for sentence in tqdm(train_data['document']):
        try:
            sentence = spell_checker.check(sentence).checked
        except Exception as check_exception:
            pass

        try:
            tokenized_sentence = okt.normalize(sentence)
            tokenized_sentence = okt.morphs(sentence, stem=True)  # Tokenizing
            stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # remove stopwords
            X_train.append(stopwords_removed_sentence)
        except Exception as morphs_exception:
            pass
    with open("X_train.pickle", "wb") as fw:
        pickle.dump(X_train, fw)

    for sentence in tqdm(test_data['document']):
        try:
            sentence = spell_checker.check(sentence).checked
        except Exception as check_exception:
            pass
        try:
            tokenized_sentence = okt.normalize(sentence)
            tokenized_sentence = okt.morphs(sentence, stem=True)  # 토큰화
            stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]  # 불용어 제거
            X_test.append(stopwords_removed_sentence)
        except Exception as morphs_exception:
            print('morphs error', sentence)
    with open("X_test.pickle", "wb") as fw:
        pickle.dump(X_test, fw)
else:
    with open("X_train.pickle", "rb") as fr:
        X_train = pickle.load(fr)

    with open("X_test.pickle", "rb") as fr:
        X_test = pickle.load(fr)


# 7. generation embedding layer based on Sentiment vetors
print('========== Generation embedding layer based on Sentiment vetors ==========')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0
total_freq = 0
rare_freq = 0

for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    if value < threshold:
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

vocab_size = total_cnt - rare_cnt + 1
print('vocab_size:', vocab_size)

max_len = 30
max_len_2 = 30

print('size of sigmoid_my_dic_data:', len(sigmoid_my_dic_sorted))
DIM_SIZE = 1
embedding_matrix_2 = np.zeros((len(sigmoid_my_dic_sorted) + 1, DIM_SIZE))

embedding_map = {}  # ('아쉽다', 1), ('재밌다', 2)

i = 0
dic_max = len(sigmoid_my_dic_sorted)
corpus = []
for sid_dic in sigmoid_my_dic_sorted:
    i += 1

    if i <= dic_max:
        for j in range(DIM_SIZE):
            embedding_matrix_2[i][j] = sid_dic[1]
        embedding_map[sid_dic[0]] = i

        corpus.append(sid_dic[0])


vocab_size_2 = len(embedding_matrix_2)

tokenizer_2 = Tokenizer(vocab_size_2)
tokenizer_2.fit_on_texts(corpus)

with open('tokenizer_info_2.txt', 'w', encoding='UTF-8') as f:
    for code, name in tokenizer_2.word_index.items():
        f.write(f'{code} : {name}\n')

X_train_2 = tokenizer_2.texts_to_sequences(X_train)
X_test_2 = tokenizer_2.texts_to_sequences(X_test)

y_train_2 = np.array(train_data['label'])
y_test_2 = np.array(test_data['label'])

X_train_2 = pad_sequences(X_train_2, maxlen=max_len_2)
X_test_2 = pad_sequences(X_test_2, maxlen=max_len_2)


tokenizer = Tokenizer(vocab_size)
tokenizer.fit_on_texts(X_train)

with open('tokenizer_info.txt', 'w', encoding='UTF-8') as f:
    for code, name in tokenizer.word_index.items():
        f.write(f'{code} : {name}\n')

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)


# Word2Vec
W2V_SIZE = 300
W2V_WINDOW = 5
W2V_EPOCH = 32
W2V_MIN_COUNT = 5

embedding_name = 'Word2Vec'
documents = [_text.split() for _text in train_data.document]
w2v_model = gensim.models.word2vec.Word2Vec(sentences=documents,
                                            size=W2V_SIZE,
                                            window=W2V_WINDOW,
                                            min_count=W2V_MIN_COUNT,
                                            # min_count=1,
                                            workers=8,
                                            sg=1)

# Embedding layer
embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        if i < vocab_size:
            embedding_matrix[i] = w2v_model.wv[word]


# 8. train
print('========== Training ==========')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

input_1 = Input(shape=(max_len,))
input_2 = Input(shape=(max_len_2,))
dense_1 = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=max_len, trainable=False)(input_1)
lstm_1 = Bidirectional(LSTM(128))(dense_1)
dense_2 = Embedding(vocab_size_2, DIM_SIZE, weights=[embedding_matrix_2], input_length=max_len_2, trainable=False)(input_2)
lstm_2 = Bidirectional(LSTM(128, activation='sigmoid'))(dense_2)
concat = Concatenate()([lstm_1, lstm_2])
outputs = Dense(100)(concat)
outputs = Dense(1, activation='sigmoid')(outputs)
model = Model(inputs=[input_1, input_2], outputs=outputs)
model.summary()

tf.keras.utils.plot_model(model,
                          to_file='model_plot.png',
                          show_shapes=True
                          )

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

history = model.fit([X_train, X_train_2], y_train, epochs=30, callbacks=[es, mc], batch_size=64, validation_split=0.2)
loaded_model = load_model('best_model.h5')

# 9. Test
print('========== Test ==========')
print("Test Accuracy: %.4f" % (loaded_model.evaluate([X_test, X_test_2], y_test)[1]))

test_text = test_data['document']
new_predict = loaded_model.predict([X_test, X_test_2])

df_score_new = pd.DataFrame(columns=['text', 'y_predict', 'labels'])

for input_text, p, label in zip(test_text, new_predict, y_test):
    df_score_new = df_score_new.append(
                pd.Series([input_text, p, label],
                          index=df_score_new.columns), ignore_index=True)
df_score_new.to_csv('results.csv', encoding="utf-8-sig")





