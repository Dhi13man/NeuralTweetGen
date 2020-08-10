import tweepy
import numpy as np
from random import randint
from time import sleep
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, TimeDistributed
from tensorflow.keras.models import Sequential, load_model


def authenticate():
    # Authenticate to Twitter
    do_it = 'y'
    while do_it == 'y':
        file = open('APIKeys.txt', 'r')
        client_key, client_secret, api_key, api_secret = file.readlines()
        print('Authenticating...')
        auth = tweepy.OAuthHandler(client_key, client_secret)
        auth.set_access_token(api_key, api_secret)
        received_api = tweepy.API(auth)  # test authentication
        try:
            received_api.verify_credentials()
            print("Authentication OK")
            return received_api
        except:
            print("Error during authentication")
            do_it = input('Retry? (y/n): ')
    return None


def get_recent_tweets(num):
    tw_json = api.search(q="hot pockets snowden", lang="en", rpp=num)
    return tw_json


def get_maps(sentences):
    # Create Mappings
    words = np.array([])
    for tweet in sentences:
        changed_tweet = tweet.replace('\n', ' ')
        tweet_words = [word for word in changed_tweet.split(' ')]
        words = np.append(words, tweet_words)
    words = sorted(set(words))

    word_ind = dict((w, ind) for ind, w in enumerate(words))
    ind_word = dict((ind, w) for ind, w in enumerate(words))

    vocabulary_size = len(word_ind)
    return word_ind, ind_word, vocabulary_size


def ready_model(sentences, batch_size, num_steps):
    word2ind, ind2word, vocabulary_size = get_maps(sentences)

    def rnn_model():
        elmo = load_model('elmo_3')
        nn = Sequential([
            Embedding(
                input_dim=vocabulary_size + 1,
                weights=elmo,
                input_length=num_steps,
                output_dim=512,
            ),
            LSTM(256,
                 return_sequences=True,
                 recurrent_initializer='glorot_uniform'
                 ),
            LSTM(256,
                 return_sequences=True,
                 recurrent_initializer='glorot_uniform'
                 ),
            Dropout(0.3),
            LSTM(256,
                 return_sequences=True,
                 recurrent_initializer='glorot_uniform'
                 ),
            TimeDistributed(Dense(vocabulary_size, activation='softmax'))
        ])
        return nn

    def get_train_data():
        x, y = np.zeros((batch_size, num_steps)), np.zeros((batch_size, num_steps, vocabulary_size))
        curr = 0
        while True:
            for num in range(batch_size):
                if curr + num_steps >= len(words):
                    # reset the index back to the start of the data set
                    curr = 0
                x[num, :] = words[curr:curr + num_steps]
                temp_y = words[curr + 1:curr + num_steps + 1]
                y[num, :, :] = to_categorical(temp_y, num_classes=vocabulary_size)
                curr += 1
            yield x, y

    # x, y = get_train_data()
    app_sentences = ''
    for tweet in sentences:
        app_sentences += tweet + ' '
    app_sentences = app_sentences.replace('\n', ' ')
    words = app_sentences.split(' ')
    words = np.array([word2ind[this_word] for this_word in words])

    model = rnn_model()

    saver_callback = ModelCheckpoint(filepath='nn_checkpoints' + '/model-{epoch:02d}.hdf5', verbose=1)
    end_callback = EarlyStopping(monitor='categorical_accuracy', min_delta=0.001)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(get_train_data(), epochs=10, steps_per_epoch=5000, batch_size=len(words) // (batch_size * num_steps),
              use_multiprocessing=True, workers=-1, callbacks=[saver_callback, end_callback])
    return model, word2ind, ind2word, vocabulary_size, num_steps


def tweet_generator(model, w_i, i_w, batch_size, total_words=15):
    batch_size = int(batch_size)
    changed_tweets = []
    for num, tweet in enumerate(tweets):
        changed_tweets.append(tweet.replace('\n', ' '))
    batches = int(total_words/batch_size)

    outs, out_word = '', ''
    x = np.zeros((batches, batch_size))

    words = changed_tweets[randint(0, len(tweets) - 1)].split(' ')
    out_word = words[randint(0, len(words) - 1)]
    for num in range(batches):
        for batch in range(batch_size):
            chosen_index = w_i[out_word]
            x[num, batch] = chosen_index
            out_word = np.argmax(model.predict(x)[num, batch, :])
            out_word = i_w[out_word]
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    api = authenticate()

    tweets = [
        'Whole semester is to be conducted online' +
        '\nBut NITians are to pay FULL TUITION FEE amount of ₹62500' +
        '\nIs this justified?' +
        '\nWe request @DrRPNishank @rashtrapatibhvn to advise the council of NITs to #ReduceNITtuitionfees under Section 32(2)(b) of NIT Act,2007' +
        '\n#ReduceReimburse' +
        '\n#COVID19',

        'We urge @TheQuint @ttindia @fayedsouza @aajtak @indiatv @ZeeNewsEnglish @BBCIndia @PragNews @DY365 to help us raise our issues regarding #ReduceReimburse #ReduceNITtuitionfees due to #COVID19' +
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
        '\n#ReduceReimburse  #ReduceNITtuitionfees @ndtv @IndiaToday @ABPNews @VICENews @ZeeNewsIndiaNo1 @rashtrapatibhvn @timesofindia @CNNnews18 @ndtv @thewire_in @ShashiTharoor @RahulGandhi @fayedsouza @BDUTT @AbhineetMishraa',

        'Due to the ongoing pandemic thousands of people have lost their livelihood' +
        '\nAnd amidst this pandemic how can you justify that the students of NIT have to pay 62000 rupees as tuition fee that too for an online semester' +
        '\nPlease help us out' +
        '\n#ReduceNITtuitionfees' +
        '\n#ReduceReimburse',

        '#ReduceNITtuitionfees' +
        '\n#ReduceReimburse' +
        '\nThis pandemic has affected everyone physically, mentally as well as economically' +
        '\nMany NITians are not in a position to pay the full 62k tuition fees as their parents have lost their jobs and are facing an economic crisis'
    ]

    # out_model, words_to_indices, indices_to_words, voc_siz, max_tok = ready_model(tweets, batch_size=6, num_steps=5)
    # out_model.save('net_speak')
    words_to_indices, indices_to_words, voc_siz = get_maps(tweets)
    out_model = load_model('nn_checkpoints\\model-02.hdf5')

    for i in range(5, 3000):
        gen = tweet_generator(out_model, words_to_indices, indices_to_words, batch_size=5, total_words=25)\
              + ' ' + str(i)
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

    # list_recieved = get_recent_tweets(3)
