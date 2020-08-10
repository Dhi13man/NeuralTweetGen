from collections import Counter

import tweepy
import numpy as np
from random import randint
from time import sleep
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential, load_model


def authenticate(api_file_name='ActualAPI.txt'):
    """
    Authenticates Twitter Developer account and returns a Tweepy based API
    :param api_file_name: String, Name of file where API keys are stored as 4 consecutive lines of no other characters:
        Line 1: Client Key
        Line 2: Client Secret
        Line 3: Api Key
        Line 4: Api Secret
    :return: Authenticated Tweepy API object
    """
    # Authenticate to Twitter
    do_it = 'y'
    while do_it == 'y':
        # Write actual keys of your Twitter Developer Account in APIKeys.txt
        # (client_key, client_secret, api_key, api_secret) in 4 consecutive lines respectively; no other characters
        file = open(api_file_name, 'r')
        temp_keys = file.readlines()
        keys = []
        for n, key in enumerate(temp_keys):
            keys.append(key[:-1] if n < len(temp_keys) - 1 else key)
        client_key, client_secret, api_key, api_secret = keys

        # Authenticate to Twitter Developer Account
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


def get_recent_tweets(num, keys, source='disk'):
    """
    Retrieves the recent most relevant tweets containing 'keys' from Twitter, using Twitter's search API.
    :param num: Integer, Number of Tweets to retrieve.
    :param keys: String, The query string to search for
    :param source: String, 'disk': Function retrieves 'num' tweets from 'tweets' file in working directory
                           'web': Function retrieves 'num' relevant tweets from Twitter search of 'keys'
    :return: Array of strings: List of tweets retrieved
    """
    new_tweets = []
    if source == 'web':
        tw_json = tweepy.Cursor(api.search, q=keys).items(num)
        for item in tw_json:
            new_tweets.append(str(item.text).replace('https://t.co/', ''))
        file = open('tweets', 'w', encoding='utf8')
        for this in new_tweets:
            file.write(this)
            file.write('\n')
    else:
        file = open('tweets', 'r', encoding='utf8')
        for this in file.readlines():
            new_tweets.append(this)
    return new_tweets


def get_characters(sentences):
    """
    Helper function to return all words from an array of sentences.
    :param sentences: Array of sentences, here tweets. Here, string
    :return: Array of words (array of strings)
    """
    characters = ''
    for tweet in sentences:
        characters += tweet
    return characters


def get_words(sentences):
    """
    Helper function to return all words from an array of sentences.
    :param sentences: Array of sentences, here tweets. Here, string
    :return: words: Array of words (array of strings)
    """
    words = np.array([])
    for tweet in sentences:
        changed_tweet = tweet.replace('\n', ' ')
        tweet_words = changed_tweet.split(' ')
        words = np.append(words, tweet_words)
    return words


def ready_dataset(element_type='word'):
    """
    Prepares Dataset for Training and Mappings for both Training and testing either in terms of words or characters.
    :param element_type: String, 'word' to predict one word at a time, 'char' to predict one character at a time.=
    :returns:
    dataset: Tensor; (input, target) data;
    ind2char: Dictionary; Mapping from index to character/word
    char2ind: List; Mapping from index to character/word
    per_epoch: Integer; Number of words/characters per epoch
    len(vocab): Integer; Vocabulary size. Number of unique elements.
    """
    text = get_words(tweets) if element_type == 'word' else get_characters(tweets)
    vocab = sorted(set(text))

    # Make Mappings
    char2ind = {char: index for index, char in enumerate(vocab)}
    ind2char = np.array(vocab)
    per_epoch = len(text) // (sequence_length + 1)

    def xy_split(this_text):
        x_text = this_text[:-1]
        y_text = this_text[1:]
        return x_text, y_text

    # Create training dataset.
    text_int = np.array([char2ind[char] for char in text])
    char_dataset = tf.data.Dataset.from_tensor_slices(text_int)
    sequences = char_dataset.batch(sequence_length + 1, drop_remainder=True)
    dataset = sequences.map(xy_split)
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    return dataset, ind2char, char2ind, per_epoch, len(vocab)


def ready_model(dataset, voc_size, batches=5, emb_dim=1, name='model'):
    """
    Creates and trains the Model on the dataset created on and passed as parameter.
    :param dataset: Tensor; (input, target) pairs prepared using ready_dataset()
    :param voc_size: Integer; Unique elements
    :param batches: Integer; Number of batches
    :param emb_dim: Integer; Depth of Embedding to be used in Embedding Keras Layer
    :param name: String; Name of models to save the checkpoints as (saved every epoch by default).
    :return: model: Tensorflow Keras Sequential RNN model trained on dataset
    """
    def rnn_model():
        nn = Sequential([
            Embedding(
                input_dim=voc_size,
                output_dim=emb_dim,
                batch_input_shape=[batches, None]
            ),
            LSTM(64,
                 return_sequences=True,
                 stateful=True,
                 recurrent_initializer='glorot_uniform'
                 ),
            LSTM(64,
                 return_sequences=True,
                 stateful=True,
                 recurrent_initializer='glorot_uniform'
                 ),
            LSTM(64,
                 return_sequences=True,
                 stateful=True,
                 recurrent_initializer='glorot_uniform'
                 ),
            Dense(vocabulary_size, activation='softmax')
        ])
        return nn

    # Prepare model and Callbacks
    model = rnn_model()
    saver_callback = ModelCheckpoint(filepath='nn_checkpoints' + '/' + name + '-{epoch:02d}.hdf5', verbose=1)
    # end_callback = EarlyStopping(monitor='categorical_accuracy', min_delta=0.001)

    # Finalise and View Model Details
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(dataset, epochs=100, use_multiprocessing=True, workers=-1, callbacks=[saver_callback])
    model.summary()
    return model


def tweet_generator(model, char2index, index2char, start_string, num_generate=1000, element_type='word', temperature=1.0):
    # Converting our start string to numbers
    input_indices = [char2index[s] for s in (start_string.split(' ') if element_type == 'word' else start_string)]
    input_indices = tf.expand_dims(input_indices, 0)

    # Empty string to store our results.
    text_generated = []

    model.reset_states()
    for char_index in range(num_generate):
        predictions = model(input_indices)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # Using a categorical distribution to predict the character returned by the model.
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions,
            num_samples=1
        )[-1, 0].numpy()

        # We pass the predicted character as the next input to the model along with the previous hidden state.
        input_indices = tf.expand_dims([predicted_id], 0)

        # Add to output
        text_generated.append((' ' + index2char[predicted_id]) if element_type == 'word' else index2char[predicted_id])

    return start_string + ''.join(text_generated)


def start_sending_tweets(run_with_api, num_words=None, model_name='model_tweet_file', element_type='word'):
    model = load_model('nn_checkpoints\\' + model_name + '.hdf5')
    # Generate and Send Tweet
    try:
        # Prepare random sequence to start with
        words = get_words(tweets)
        index = randint(0, len(words) - 1 - sequence_length)
        words = words[index:index + sequence_length]
        start = ''
        for word in words:
            start += word + ' '

        gen = tweet_generator(model, c2i, i2c, start, num_generate=num_words, element_type=element_type) + ' '
        print('GENERATED: %s' % gen)
        print('Sending Tweet', end=': a, ')
        run_with_api.update_status(gen)
        sleep(randint(30, 50))
        print('b', end=', ')
        run_with_api.update_status(tweets[1])
        sleep(randint(30, 50))
        print('c', end=', ')
        run_with_api.update_status(tweets[2])
        sleep(randint(50, 90))
        print('d', end=', ')
        run_with_api.update_status(tweets[3])
        sleep(randint(50, 90))
        print('e', end=', ')
        run_with_api.update_status(tweets[0])
        sleep(randint(70, 90))
        print('f', end=', ')
        run_with_api.update_status(tweets[4])
        sleep(randint(70, 90))
        print('g')
        run_with_api.update_status(tweets[5])
        sleep(randint(90, 110))
    except tweepy.error.TweepError:
        choice = input('Error while Sending. Try Again? (Y/N): ')
        if choice == 'Y':
            start_sending_tweets()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Authenticate and analyse Basic Global Variables for RNN
    api = authenticate()
    embedding_dimensions, batch_size, buffer_size, sequence_length, e_type = 10, 1, 10000, 15, 'char'

    # List of central tweets to revolve your Automatically Generated Tweets around
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

    # Use extra tweets in tweets file or the web (searched with given keywords) for training
    words_in_tweets, keywords = Counter(get_words(tweets)), ''
    for common_word in words_in_tweets.most_common(3):
        keywords += common_word[0] + ' '
    new = get_recent_tweets(1000, keywords, 'disk')
    for tweet_this in new:
        tweets.append(tweet_this)

    # Get Data and Train
    data, i2c, c2i, text_per_epoch, vocabulary_size = ready_dataset(e_type)
    out_model = ready_model(data, voc_size=vocabulary_size, batches=batch_size, emb_dim=embedding_dimensions)

    # Use trained Models
    start_sending_tweets(api, num_words=25, model_name='model-64', element_type=e_type)
