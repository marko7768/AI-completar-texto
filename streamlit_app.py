import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
import joblib

model = load_model('modelo_text.h5')
tokenizer = joblib.load('tokenizer.pkl')

st.title('Aplicación ML')

input_text = st.text_input('Ingrese texto a completar')

def generate_text(model, tokenizer, input_text, num_words):
sequences = tokenizer.texts_to_sequences(texts[0])

# Padding para igualar la longitud de las secuencias
maxlen = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=maxlen)
 
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