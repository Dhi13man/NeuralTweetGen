from collections import defaultdict

import tweepy
import numpy as np
from random import randint
from time import sleep
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, TimeDistributed
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.initializers import Constant


def authenticate():
    # Authenticate to Twitter
    do_it = 'y'
    while do_it == 'y':
        # Write actual keys of your Twitter Developer Account in APIKeys.txt
        # (client_key, client_secret, api_key, api_secret) in 4 consecutive lines respectively; no other characters
        file = open('APIKeys.txt', 'r')
        temp_keys = file.readlines()
        keys = []
        for n, key in enumerate(temp_keys):
            keys.append(key[:-1] if n < len(temp_keys) - 1 else key)
        client_key, client_secret, api_key, api_secret = keys
        print('Authenticating...')
        auth = tweepy.OAuthHandler(client_key, client_secret)
        auth.set_access_token(api_key, api_secret)
        received_api = tweepy.API(auth)  # test authentication
        try:
            received_api.verify_credentials()
            print("Authentication OK")
            return received_api
        except tweepy.TweepError:
            print("Error during authentication")
            do_it = input('Retry? (y/n): ')
    return None


def get_recent_tweets(num):
    tw_json = api.search(q="hot pockets snowden", lang="en", rpp=num)
    return tw_json


def get_words(sentences):
    words = np.array([])
    for tweet in sentences:
        changed_tweet = tweet.replace('\n', ' ')
        tweet_words = changed_tweet.split(' ')
        words = np.append(words, tweet_words)
    return words


def get_maps(sentences):
    # Create Mappings
    words = sorted(set(get_words(sentences)))
    word_ind = dict((w, ind) for ind, w in enumerate(words))
    ind_word = dict((ind, w) for ind, w in enumerate(words))

    vocabulary_size = len(word_ind)
    return word_ind, ind_word, vocabulary_size


def use_glove_emb(sentences, num_dim=25):
    if num_dim not in [25, 50, 100, 200]:
        emb_dim = 25
    else:
        emb_dim = num_dim

    def load_embedding_from_disks(glove_filename):
        word_to_index_dict = dict()
        index_to_embedding_array = []

        with open(glove_filename, 'r', encoding="utf8") as glove_file:
            for (i, line) in enumerate(glove_file.readlines()):
                split = line.split(' ')
                word = split[0]

                representation = split[1:]
                representation = np.array(
                    [float(val) for val in representation]
                )

                word_to_index_dict[word] = i
                index_to_embedding_array.append(representation)

        _WORD_NOT_FOUND = [0.0] * len(representation)  # Empty representation for unknown words
        _LAST_INDEX = i + 1
        word_to_index_dict = defaultdict(lambda: _LAST_INDEX, word_to_index_dict)
        index_to_embedding_array = np.array(index_to_embedding_array + [_WORD_NOT_FOUND])
        return word_to_index_dict, index_to_embedding_array

    w0, i0 = load_embedding_from_disks('glove' + str(emb_dim) + 'd.txt')

    words = sorted(set(get_words(sentences)))
    i0 = i0[:, :num_dim]

    # Add unknown words present in Sentences to Embeddings and indices
    for word in words:
        if word not in list(w0.keys()):
            temp = int(np.max(i0) + 1) * (2 * np.random.rand(1, num_dim) - 1)
            w0.__setitem__(word, i0.shape[0])
            i0 = np.concatenate([i0, temp], axis=0)
    del temp

    # Create required dictionaries and Embedding Matrix
    word2ind, ind2emb = w0, i0
    ind2word_emb = [dict(zip(word2ind.values(), word2ind.keys())), ind2emb]

    return word2ind, ind2emb, ind2word_emb


def ready_model(sentences, batches=5, emb_dim=1, step_size=1, name='model'):
    if name == 'glove':
        word2ind, ind2emb, ind2word = use_glove_emb(sentences, num_dim=emb_dim)
        vocabulary_size = len(word2ind) + 1
    else:
        word2ind, ind2word, vocabulary_size = get_maps(sentences)

    def rnn_model():
        nn = Sequential([
            Embedding(
                input_dim=vocabulary_size,
                embeddings_initializer=Constant(ind2emb) if name == 'glove' else None,
                trainable=False,
                output_dim=emb_dim,
            ),
            LSTM(20,
                 return_sequences=True,
                 recurrent_initializer='glorot_uniform'
                 ),
            LSTM(20,
                 return_sequences=True,
                 recurrent_initializer='glorot_uniform'
                 ),
            TimeDistributed(Dense(emb_dim))
        ])
        return nn

    def get_train_data():
        x, y = np.zeros((batches, step_size)), np.zeros((batches, step_size, emb_dim))
        curr = 0
        while True:
            for num in range(batches):
                if curr + step_size >= len(words):
                    # reset the index back to the start of the data set
                    curr = 0
                x[num, :] = words[curr:curr + step_size]
                for step in range(step_size):
                    y[num, step, :] = ind2emb[words[curr + step + 1]]
                curr += 1
            yield x, y

    words = get_words(sentences)
    words = np.array([word2ind[this_word] for this_word in words])

    model = rnn_model()
    saver_callback = ModelCheckpoint(filepath='nn_checkpoints' + '/' + name + '-{epoch:02d}.hdf5', verbose=1)
    # end_callback = EarlyStopping(monitor='categorical_accuracy', min_delta=0.001)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(get_train_data(), epochs=10, steps_per_epoch=5000, batch_size=len(words) // (batches * step_size),
              use_multiprocessing=True, workers=-1, callbacks=[saver_callback])
    model.summary()
    return model, word2ind, ind2word, vocabulary_size


def tweet_generator(model, w_i, i_w, num_steps=1, total_words=15):
    # Pre-Computations
    changed_tweets = []
    for num, tweet in enumerate(tweets):
        changed_tweets.append(tweet.replace('\n', ' '))
    batches = int(total_words / num_steps)
    if embedding_type == 'glove':
        ind2word = i_w[0]
        ind2emb = i_w[1]

    outs, out_word = '', ''

    words = changed_tweets[randint(0, len(tweets) - 1)].split(' ')
    out_word = words[randint(0, len(words) - 1)]
    x = np.zeros(num_steps)

    for batch in range(batches):
        for step in range(num_steps):
            chosen_index = w_i[out_word]
            x[step] = chosen_index
            out_emb = model.predict(x)[step, :]
            closest_emb_index = np.argmax(np.sum((ind2emb - out_emb) ** 2, axis=1))
            out_word = ind2word[closest_emb_index]
            outs += out_word + ' '
        
    # Formatting Tweet Better
    final_string, flag = '', 0
    for out_num, char in enumerate(outs):
        if flag == 0 or not char.isalpha():
            final_string += char
        else:
            final_string += char.upper()
            flag = 0
            continue
        if not char.isalnum():
            flag = 1
    outs = final_string.capitalize()
    return outs


def start_sending_tweets(api=None, num_words=None, name='model'):
    if num_words is None:
        num_words = Batches * step_size

    # Get required mappings
    words_to_indices, indices_to_words, _ = get_maps(tweets)
    if name == 'glove':
        words_to_indices, indices_to_emb, indices_to_words = use_glove_emb(tweets, embedding_dimensions)

    model1 = load_model('nn_checkpoints\\' + name + '-06.hdf5')
    model2 = load_model('nn_checkpoints\\' + name + '-06.hdf5')
    for i in range(10, 3000):
        try:
            model = model1 if i % 2 == 0 else model2
            gen = tweet_generator(model, words_to_indices, indices_to_words, num_steps=step_size
                                  , total_words=num_words) + ' ' + str(i)
            print('GENERATED: %s' % gen)
            print('Sending Tweet %d' % i, end=': a, ')
            api.update_status(gen)
            sleep(randint(30, 50))
            print('b', end=', ')
            api.update_status(tweets[1] + ' ' + str(i))
            sleep(randint(30, 50))
            print('c', end=', ')
            api.update_status(tweets[2] + ' ' + str(i))
            sleep(randint(50, 90))
            print('d', end=', ')
            api.update_status(tweets[3] + ' ' + str(i))
            sleep(randint(50, 90))
            print('e', end=', ')
            api.update_status(tweets[0] + ' ' + str(i))
            sleep(randint(70, 90))
            print('f', end=', ')
            api.update_status(tweets[4] + ' ' + str(i))
            sleep(randint(70, 90))
            print('g')
            api.update_status(tweets[5] + ' ' + str(i))
            sleep(randint(90, 110))
        except tweepy.error.TweepError:
            continue


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    api = authenticate()
    embedding_dimensions, Batches, step_size, embedding_type = 10, 5, 5, 'glove'

    tweets = [
        'Whole semester is to be conducted online' +
        '\nBut NITians are to pay FULL TUITION FEE amount of ₹62500' +
        '\nIs this justified?' +
        '\nWe request @DrRPNishank @rashtrapatibhvn to advise the council of NITs to #ReduceNITtuitionfees under '
        'Section 32(2)(b) of NIT Act,2007' +
        '\n#ReduceReimburse' +
        '\n#COVID19',

        'We urge @TheQuint @ttindia @fayedsouza @aajtak @indiatv @ZeeNewsEnglish @BBCIndia @PragNews @DY365 to help '
        'us raise our issues regarding #ReduceReimburse #ReduceNITtuitionfees due to #COVID19' +
        '\n@HRDMinistry @DrRPNishank @PMOIndia @rashtrapatibhvn' +
        '\n#mhrd',

        'Online classes demands hefty internet expenses' +
        '\nWe request the NITs to REIMBURSE A PART OF THE MESS EXCESS to cover the internet expenses of #NITians' +
        '\n@HRDMinistry @DrRPNishank' +
        '\n@PMOIndia' +
        '\n#ReduceReimburse' +
        '\n#ReduceNITtuitionfees' +
        '\n#mhrd' +
        '\n#COVID19',

        '62.5k tuition fees+Many other charges by NITs across India' +
        '\n#ReduceReimburse  #ReduceNITtuitionfees @ndtv @IndiaToday @ABPNews @VICENews @ZeeNewsIndiaNo1 '
        '@rashtrapatibhvn @timesofindia @CNNnews18 @ndtv @thewire_in @ShashiTharoor @RahulGandhi @fayedsouza @BDUTT '
        '@AbhineetMishraa',

        'Due to the ongoing pandemic thousands of people have lost their livelihood' +
        '\nAnd amidst this pandemic how can you justify that the students of NIT have to pay 62000 rupees as tuition '
        'fee that too for an online semester' +
        '\nPlease help us out' +
        '\n#ReduceNITtuitionfees' +
        '\n#ReduceReimburse',

        '#ReduceNITtuitionfees' +
        '\n#ReduceReimburse' +
        '\nThis pandemic has affected everyone physically, mentally as well as economically' +
        '\nMany NITians are not in a position to pay the full 62k tuition fees as their parents have lost their jobs '
        'and are facing an economic crisis '
    ]

    out_model, words_to_indices, indices_to_words, voc_siz = ready_model(tweets, batches=Batches, step_size=step_size,
                                                                         emb_dim=embedding_dimensions,
                                                                         name=embedding_type)
    out_model.save('net_speak')

    start_sending_tweets(name=embedding_type, num_words=25)
