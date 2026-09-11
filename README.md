# NeuralTweetGen

Train a Keras LSTM on a small tweet corpus and sample new tweet-like text. The project is a character- and word-level recurrent neural network experiment, not a hosted tweet bot.

## Install

Python 3 with TensorFlow/Keras, NumPy, and (optional) Tweepy if you want to post samples.

```sh
git clone https://github.com/Dhi13man/NeuralTweetGen.git
cd NeuralTweetGen
pip install tensorflow numpy tweepy
```

Put Twitter API credentials in a local file that is **not** committed. Do not put keys in the repository.

## Use

- Training and sampling live in `main.py` (LSTM embedding, dropout, checkpoints under `nn_checkpoints/`).
- Seed text lives under `tweets/`.
- Saved models include character- and word-level HDF5 checkpoints.

```sh
python main.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
