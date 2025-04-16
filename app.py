import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import pickle

# ===========================
# Device Setting
# ===========================
if torch.cuda.is_available():
    device = torch.device("cuda")
    st.write("Using NVIDIA GPU (CUDA)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    st.write("Using Mac GPU (MPS)")
else:
    device = torch.device("cpu")
    st.write("Using CPU")


# ===========================
# Special Tokens and Vocabulary for LSTM Model
# ===========================

# Load the LSTM vocabulary from the saved pickle file
vocab_path = "saved_models/lstm/vocab.pkl"
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)

# Create reverse vocabulary for decoding token ids back to words.
reverse_vocab = {idx: word for word, idx in vocab.items()}

# These tokens must match the training configuration.
PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<PAD>", "<SOS>", "<EOS>", "<UNK>"

MAX_ARTICLE_LEN = 400   # Same as used during training.
MAX_SUMMARY_LEN = 100

def text_to_sequence(text, max_len, add_eos=False):
    text = text.lower()
    words = text.split()
    if add_eos:
        words.append(EOS_TOKEN)
    # Truncate and map words to indices
    words = words[:max_len]
    seq = [vocab.get(word, vocab[UNK_TOKEN]) for word in words]
    # Pad if needed.
    if len(seq) < max_len:
        seq.extend([vocab[PAD_TOKEN]] * (max_len - len(seq)))
    return seq


# ===========================
# LSTM Model Components (Encoder, Attention, Decoder)
# ===========================
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab[PAD_TOKEN])
        self.lstm = nn.LSTM(embed_dim, enc_hidden_dim, batch_first=True)
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, attn_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(enc_hidden_dim + dec_hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)
    def forward(self, dec_hidden, enc_outputs):
        batch_size, src_len, _ = enc_outputs.size()
        dec_hidden_expanded = dec_hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((enc_outputs, dec_hidden_expanded), dim=2)))
        attention = self.v(energy).squeeze(2)
        attn_weights = torch.softmax(attention, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_hidden_dim, dec_hidden_dim, attention):
        super(Decoder, self).__init__()
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab[PAD_TOKEN])
        self.lstm = nn.LSTM(enc_hidden_dim + embed_dim, dec_hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(enc_hidden_dim + dec_hidden_dim + embed_dim, vocab_size)
    def forward(self, input_token, hidden, cell, enc_outputs):
        input_token = input_token.unsqueeze(1)
        embedded = self.embedding(input_token)
        dec_hidden_last = hidden[-1]
        context, _ = self.attention(dec_hidden_last, enc_outputs)
        context = context.unsqueeze(1)
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)
        pred = self.fc_out(torch.cat((output, context, embedded), dim=1))
        return pred, hidden, cell

def generate_summary_lstm(article, encoder, decoder):
    encoder.eval()
    decoder.eval()
    # Convert input article into a padded sequence.
    seq = text_to_sequence(article, MAX_ARTICLE_LEN, add_eos=False)
    src_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        enc_outputs, hidden, cell = encoder(src_tensor)
        dec_input = torch.tensor([vocab[SOS_TOKEN]], dtype=torch.long).to(device)
        summary_tokens = []
        for _ in range(MAX_SUMMARY_LEN):
            output, hidden, cell = decoder(dec_input, hidden, cell, enc_outputs)
            pred_token = output.argmax(1)
            if pred_token.item() == vocab[EOS_TOKEN]:
                break
            summary_tokens.append(pred_token.item())
            dec_input = pred_token
    summary_words = [reverse_vocab.get(idx, UNK_TOKEN) for idx in summary_tokens]
    summary = " ".join(summary_words)
    return summary


# ===========================
# Model Loading Functions (with caching)
# ===========================
@st.cache_resource
def load_lstm_model():
    vocab_size = len(vocab)
    embed_dim = 128
    enc_hidden_dim = 256
    dec_hidden_dim = 256
    attn_dim = 256
    encoder_model = Encoder(vocab_size, embed_dim, enc_hidden_dim)
    attention_model = Attention(enc_hidden_dim, dec_hidden_dim, attn_dim)
    decoder_model = Decoder(vocab_size, embed_dim, enc_hidden_dim, dec_hidden_dim, attention_model)
    checkpoint = torch.load("saved_models/lstm/best_model.pth", map_location=device)
    encoder_model.load_state_dict(checkpoint['encoder_state_dict'])
    decoder_model.load_state_dict(checkpoint['decoder_state_dict'])
    encoder_model.to(device)
    decoder_model.to(device)
    return encoder_model, decoder_model

@st.cache_resource
def load_t5_model():
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("saved_models/t5-small-lora/best_model")
    model.to(device)
    return model, tokenizer

@st.cache_resource
def load_bart_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("saved_models/bart-large-cnn-lora/best_model")
    model.to(device)
    return model, tokenizer

# Simple text cleaning function.
def preprocess_text(text):
    return text.strip()

def generate_summary_t5(article, model, tokenizer):
    input_text = "summarize: " + article
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    summary_ids = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=128)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def generate_summary_bart(article, model, tokenizer):
    inputs = tokenizer(article, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=128)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# ===========================
# Streamlit App Layout
# ===========================
st.title("News Summarization App")
st.write("Paste your news article below, select the model, and click the button to generate a summary.")

# Multi-line text box for input.
user_input = st.text_area("Enter News Article", height=300)

# Model selection: LSTM with attention, T5-small (LoRA), or BART-large-cnn (LoRA)
model_choice = st.selectbox("Select Model", options=["LSTM", "Fine-tuned T5", "Fine-tuned BART"])


if st.button("Generate Summary"):
    if not user_input.strip():
        st.warning("Please enter some text to summarize.")
    else:
        article = preprocess_text(user_input)
        summary = ""
        with st.spinner("Generating summary..."):
            if model_choice == "LSTM":
                encoder_model, decoder_model = load_lstm_model()
                summary = generate_summary_lstm(article, encoder_model, decoder_model)
            elif model_choice == "Fine-tuned T5":
                model, tokenizer = load_t5_model()
                summary = generate_summary_t5(article, model, tokenizer)
            elif model_choice == "Fine-tuned BART":
                model, tokenizer = load_bart_model()
                summary = generate_summary_bart(article, model, tokenizer)
        st.success("Summary Generated!")
        st.markdown("### Generated Summary")
        st.write(summary)