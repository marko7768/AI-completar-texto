import streamlit as st
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('modelo_text.h5')
tokenizer = joblib.load('tokenizer.pkl')

st.title('Aplicación AI')

input_text = st.text_input('Ingrese texto a completar')

def generate_text(model, tokenizer, input_text, num_words):
    input_seq = tokenizer.texts_to_sequences([input_text])

    input_seq = pad_sequences(input_seq, maxlen=20)

    generated_sentence = input_text
    for _ in range(num_words):
        predicted_probabilities = model.predict(input_seq, verbose=0)
        predicted_word_idx = np.random.choice(len(predicted_probabilities[0]), p=predicted_probabilities[0])
        generated_word = tokenizer.index_word.get(predicted_word_idx, "")

        if generated_word:
            generated_sentence += " " + generated_word
            input_seq = np.append(input_seq[:, 1:], predicted_word_idx)
            input_seq = input_seq.reshape(1, -1)

    return generated_sentence

if st.button('Completar'):
    generated_text = generate_text(model, tokenizer, input_text, num_words=3)
    st.write(f'{generated_text}')