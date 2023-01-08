from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model # Add 'load_model'
from joblib import dump, load # For reading the Tokenizer Pickle
from lstm_2 import decode_sentiment, predict
KERAS_MODEL = "model.h5"
TOKENIZER_MODEL = "tokenizer.pkl"

# KERAS
SEQUENCE_LENGTH = 300

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# Load the model and the tokenizer to make predictions
model = load_model(KERAS_MODEL)
tokenizer = load(TOKENIZER_MODEL)
decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
predict("I love the music")
