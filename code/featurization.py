import pandas as pd
import numpy as np
import scipy.sparse as sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import conf
try: import cPickle as pickle   # python2
except: import pickle           # python3

np.set_printoptions(suppress=True)

import sys
try: #python2
    reload(sys)
    sys.setdefaultencoding('utf-8')
except: pass

train_input = conf.train_tsv
test_input = conf.test_tsv
train_output = conf.train_matrix
test_output = conf.test_matrix

def get_df(input):
    df = pd.read_csv(
        input,
        encoding='utf-8',
        header=None,
        delimiter='\t',
        names=['id', 'label', 'text']
    )
    sys.stderr.write('The input data frame {} size is {}\n'.format(input, df.shape))
    return df

def save_matrix(df, matrix, output):
    id_matrix = sparse.csr_matrix(df.id.astype(np.int64)).T
    label_matrix = sparse.csr_matrix(df.label.astype(np.int64)).T

    result = sparse.hstack([id_matrix, label_matrix, matrix], format='csr')

    msg = 'The output matrix {} size is {} and data type is {}\n'
    sys.stderr.write(msg.format(output, result.shape, result.dtype))

    with open(output, 'wb') as fd:
        pickle.dump(result, fd, pickle.HIGHEST_PROTOCOL)
    pass

df_train = get_df(train_input)
print(df_train.shape)
# print(df_train.columns)
print(df_train.dtypes)
print('step1')
# train_words = np.array(df_train.text.str.lower().values.astype('U'))
train_words = df_train.text.str.lower().values.astype('U')
print(train_words.shape, train_words.dtype)

print('step2')
bag_of_words = CountVectorizer(stop_words='english',
                               max_features=5000)
print('step3')
bag_of_words.fit(train_words)
print('step4')
train_words_binary_matrix = bag_of_words.transform(train_words)

print('step5')
tfidf = TfidfTransformer(smooth_idf=False)
print('step6')
tfidf.fit(train_words_binary_matrix)
print('step7')
train_words_tfidf_matrix = tfidf.transform(train_words_binary_matrix)
print('step8')
save_matrix(df_train, train_words_tfidf_matrix, train_output)
del df_train

df_test = get_df(test_input)
print('step2-1')
#test_words = np.array(df_test.text.str.lower().values.astype('U'))
test_words = df_test.text.str.lower().values.astype('U')
print('step2-2')
test_words_binary_matrix = bag_of_words.transform(test_words)
print('step2-3')
test_words_tfidf_matrix = tfidf.transform(test_words_binary_matrix)
print('step2-4')
save_matrix(df_test, test_words_tfidf_matrix, test_output)

