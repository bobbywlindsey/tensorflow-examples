import io
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Get a pandas DataFrame object of all the data in the csv file:
df = pd.read_csv('tweets.csv')
print(df.shape)

# Get pandas Series object of the "tweet text" column:
text = df['tweet_text']
df.rename({'is_there_an_emotion_directed_at_a_brand_or_product': 'target',
          'tweet_text': 'text'}, axis=1, inplace=True)

# Remove the blank rows from the series:
df = df[df.text.notna()]
print(df.shape)
df = df[['text', 'target']]

# Convert the string label to a 0 or 1
df['target'] = df.target.apply(lambda x: 0 if x == 'Negative emotion' else 1)

print(df.head())

# Split data into train and test sets
train = df.iloc[0:8000]
test = df.iloc[8000:]
train_x = train.text
train_y = train.target
test_x = test.text
test_y = test.target

# Specify embedding params
vocab_size = 10000
sequence_length = 128 # how long a tweet is
trunc_type='post' # truncate from the end of the sentence
oov_tok = '<OOV>'
embedding_dim = 16

# Map the words in the tweets to integers
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_x)

# Here's the vocab
word_index = tokenizer.word_index
print(word_index)

# Convert tweets to integer sequences
train_sequences = tokenizer.texts_to_sequences(train_x)
test_sequences = tokenizer.texts_to_sequences(test_x)

# Pad the sequences from the right just in case they're short of 128
train_padded = pad_sequences(train_sequences, maxlen=sequence_length, truncating=trunc_type)
test_padded = pad_sequences(test_sequences, maxlen=sequence_length, truncating=trunc_type)

# Build your model
model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  input_length=sequence_length, name='embedding'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

# Train the model
model.fit(train_padded, train_y,
          epochs=10, validation_data=(test_padded, test_y))

# Test the model on new tweets
new_sents = [
    'These iPad apps really suck. Cant even use it as a desktop',
    'Wow, so great this iPhone Fantastical app',
    'Android is the most flexible OS ever'
]
new_seq = tokenizer.texts_to_sequences(new_sents)
padded = pad_sequences(new_seq, maxlen=sequence_length, truncating=trunc_type)
output = model.predict(padded)
for i in range(0,len(new_sents)):
    print('Review:'+new_sents[i]+' '+'sentiment:'+str(output[i])+'\n')
    
# Get the embedding layer
e = model.layers[0] # get the first layer (i.e. embedding layer)
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

# Here's embedding for word google
print(weights[word_index['google']])

# Write out the labels and word vectors and upload them to
# http://projector.tensorflow.org/ to display your vectors in a 3d space
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for word, num in word_index.items():
    if num == 0: continue # skip padding token from vocab
    vec = weights[num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()

try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download('vecs.tsv')
    files.download('meta.tsv')
