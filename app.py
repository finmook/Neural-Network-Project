import streamlit as st
import pandas as pd
import numpy as np

import json
import pickle
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Toxic Comment Detection",
    page_icon="💬",
    layout="wide"
)

LABELS=["non-offensive","offensive"]

TRANSFORMER_PATH="./saved_transformer"
BILSTM_MODEL_PATH="./saved_bilstm/saved_bilstm.keras"
BILSTM_TOKENIZER_PATH="./saved_bilstm/bilstm_tokenizer.pkl"
BILSTM_CONFIG_PATH="./saved_bilstm/bilstm_config.json"

@st.cache_resource
def load_transformer():
    tokenizer=AutoTokenizer.from_pretrained(TRANSFORMER_PATH)
    model=AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_PATH)
    model.eval()
    return tokenizer,model

@st.cache_resource
def load_bilstm():
    model=tf.keras.models.load_model(BILSTM_MODEL_PATH)
    with open(BILSTM_TOKENIZER_PATH,"rb") as f:
        tokenizer=pickle.load(f)

    with open(BILSTM_CONFIG_PATH,"r") as f:
        config=json.load(f)

    return model,tokenizer,config

def label_from_probability(probs):
    offensive_prob=probs[1]
    if(offensive_prob>=0.6):
        return "offensive"
    elif offensive_prob<=0.4:
        return "non-offensive"
    else:
        return "uncertain"
    
def predict_transformer(text,tokenizer,model):
    inputs=tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )
    with torch.no_grad():
        inputs.pop("token_type_ids", None)
        outputs=model(**inputs)
        probs=torch.softmax(outputs.logits,dim=1).numpy()[0]

    pred_label=label_from_probability(probs)
    return pred_label,probs

def predict_bilstm(text,model,tokenizer,config):
    max_len=config["max_len"]
    sequence=tokenizer.texts_to_sequences([text])
    padded=pad_sequences(
        sequence,
        maxlen=max_len,
        padding="post",
        truncating="post"    
    )

    prob_offensive=float(model.predict(padded, verbose=0)[0][0])#sigmoid output

    probs=np.array([
        1-prob_offensive,
        prob_offensive
    ])

    pred_label=label_from_probability(probs)
    return pred_label,probs

def show_result(model_name,pred_label,probs):
    st.subheader(model_name)
    col1,col2=st.columns(2)

    with col1:
        st.metric("Prediction",pred_label)

    with col2:
        st.metric("Offensive Probabilty",f"{probs[1]:.2%}")
    
    prob_df=pd.DataFrame({
        "Label":LABELS,
        "Probability":probs
    })

    st.bar_chart(prob_df,x="Label",y="Probability")

def clear_comment():
    st.session_state["comment_area"]=""

st.title("Toxic Comment Detection")
st.write("Enter a text for detecting offensive content")

st.sidebar.header("Settings")
model_choice=st.sidebar.selectbox(
    "Choose model",
    ["Transformer","BILSTM","Compare both"]
)

st.subheader("Input Text")

user_text=st.text_area(
    "Type a comment or tweet:",
    key="comment_area",
    height=120,
    placeholder="Example: you are so stupid"
)
col_btn1, col_btn2, _ = st.columns([2, 2, 6])

with col_btn1:
    predict_button = st.button("Predict", use_container_width=True)

with col_btn2:
    clear_button = st.button(
        "Clear",
        on_click=clear_comment,
        use_container_width=True
    )

if predict_button:
    if user_text.strip()=="":
        st.warning("Please enter some text.")
    else:
        transformer_tokenizer,transformer_model=load_transformer()
        bilstm_model, bilstm_tokenizer, bilstm_config = load_bilstm()
        if model_choice=="Transformer":
            pred_label, probs = predict_transformer(
                user_text,
                transformer_tokenizer,
                transformer_model
            )

            show_result("Transformer Result", pred_label, probs)

        elif model_choice == "BiLSTM":

            pred_label, probs = predict_bilstm(
                user_text,
                bilstm_model,
                bilstm_tokenizer,
                bilstm_config
            )

            show_result("BiLSTM Result", pred_label, probs)

        elif model_choice == "Compare both":
            col1, col2 = st.columns(2)

            with col1:
                pred_label_t, probs_t = predict_transformer(
                    user_text,
                    transformer_tokenizer,
                    transformer_model
                )

                show_result("Transformer", pred_label_t, probs_t)

            with col2:
                pred_label_b, probs_b = predict_bilstm(
                    user_text,
                    bilstm_model,
                    bilstm_tokenizer,
                    bilstm_config
                )
                show_result("BiLSTM", pred_label_b, probs_b)
        

with st.expander("How does this app work?"):
    st.write("""
    This app uses two models: BiLSTM and Transformer.

    The BiLSTM model converts text into word index sequences using the saved Keras tokenizer.
    Then, the sequence is padded and passed into the BiLSTM model to predict the offensive probability.

    The Transformer model uses a DistilBERT tokenizer to convert text into input IDs and attention masks.
    The fine-tuned Transformer outputs logits for two classes, which are converted into probabilities using softmax.

    The original dataset has only two real classes: non-offensive and offensive.
    The uncertain output is added in the UI when the offensive probability is between 0.40 and 0.60.
    This means the model is not confident enough to choose one side clearly.
    """)






