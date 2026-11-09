# NLP and Transformers

From simple bag-of-words models to Transformers, built step by step on movie reviews (sentiment) and English-to-Spanish translation.

## Scripts

### Sentiment analysis on IMDB reviews

| File | Approach |
|---|---|
| `imdb_data_download.py` | Downloads and extracts the IMDB reviews dataset (`aclImdb/`) |
| `imdb_bag_of_bigrams.py` | **Bag of words:** each review becomes a TF-IDF vector of single words and word pairs (bigrams), then a small Dense network classifies it |
| `imdb_bidirectional_lstm.py` | **Sequence model:** words become one-hot vectors in order, read by a Bidirectional LSTM |
| `transformer_encoder.py` ⭐ | **Transformer:** word embeddings, then a custom `TransformerEncoder` layer (multi-head self-attention, a feed-forward block, residual connections and layer normalization) |

### Machine translation (English to Spanish)

| File | Approach |
|---|---|
| `translation_data_prep.py` | Loads the English-Spanish sentence pairs, splits them into train, validation and test, and vectorizes them |
| `seq2seq_transformer.py` ⭐ | A full **encoder-decoder Transformer** built from scratch: `PositionalEmbedding`, `TransformerEncoder`, `TransformerDecoder` with a causal attention mask, and a decoding loop that translates new sentences word by word |

### Text generation (work in progress)

| File | Approach |
|---|---|
| `text_generation.py` | A Transformer language model trained on IMDB reviews to generate text from a prompt, with temperature sampling. **Not finished yet.** |

## The Transformer layers I implemented

```
PositionalEmbedding   = token embedding + position embedding
TransformerEncoder    = self-attention → add & norm → feed-forward → add & norm
TransformerDecoder    = masked self-attention → add & norm
                        → cross-attention to encoder → add & norm
                        → feed-forward → add & norm
```

## Data

- **IMDB:** run `imdb_data_download.py` (about 80 MB). Create the `aclImdb/val` folder by moving 20% of `train` into it.
- **English-Spanish:** download [spa-eng.zip](http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip) and extract `spa.txt` into `spa_eng/`.

## Run it

```bash
pip install tensorflow numpy
python imdb_bag_of_bigrams.py
python imdb_bidirectional_lstm.py
python transformer_encoder.py
python seq2seq_transformer.py
```

## What I learned

- The two families of NLP models: **sets** (bag of words) vs **sequences** (RNNs, Transformers), and when each one wins.
- How self-attention lets every word look at every other word, and why that replaced RNNs.
- Building custom Keras layers (`get_config` for saving and loading, masking for padding).
- How a decoder generates text one token at a time with a causal mask.

## Credits

Based on *Deep Learning with Python* by François Chollet (Manning).
