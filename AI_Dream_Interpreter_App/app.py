import nltk
import streamlit as st
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure NLTK resources are downloaded
nltk.download("vader_lexicon")
nltk.download("punkt")

# Streamlit UI Customization
st.set_page_config(page_title="AI Dream Interpreter", page_icon=" 🛌💤🌙", layout="centered")

st.image("https://media.giphy.com/media/26BRzozg4TCBXv6QU/giphy.gif", use_container_width=True)

st.markdown("### 🌙 Dream 🛌💤")

# Title and Description
st.markdown("<h1 style='text-align: center;'>🌙 AI Dream Interpreter</h1>", unsafe_allow_html=True)
st.write("💭 Select a common dream or describe your own dream for AI interpretation.")

# Cache heavy AI models
@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis")
    dream_model = pipeline("text-generation", model="gpt2")
    return sentiment_model, dream_model

sentiment_model, dream_model = load_models()

# Initialize Sentiment Analysis
sia = SentimentIntensityAnalyzer()

# Full list of dreams added for selection
dream_suggestions = [
    "Flying in the sky 🕊️",
    "Falling from a high place ⬇️",
    "Being chased by something scary 👣",
    "Seeing a snake 🐍",
    "Losing your teeth 😬",
    "Finding a lot of money 💰",
    "Being trapped in darkness 🌑",
    "Seeing yourself in a mirror 🪞",
    "Walking on water 🌊",
    "Standing on top of a mountain ⛰️",
    "Crossing a long bridge 🌉",
    "Being caught in a storm 🌩️",
    "Seeing a bright light 🌟",
    "Running out of time ⏳",
    "Holding a mysterious key 🔑",
    "Climbing a long staircase 🏛️",
    "Standing on a lonely island 🏝️",
    "Watching a fire burn 🔥",
    "Holding a baby 👶",
    "Talking to a deceased loved one 💀",
    "Driving an out-of-control car 🚗",
    "Finding an unknown door 🚪",
]

# Common Dream Symbols
dream_symbols = {
    "Flying": "You seek freedom or have high ambitions.",
    "Falling": "You might be experiencing anxiety or fear of failure.",
    "Being Chased": "May indicate avoiding something in real life.",
    "Snake": "Symbolizes transformation or hidden fears.",
    "Teeth Falling Out": "You may be worried about your appearance or communication.",
    "Finding Money": "Represents financial luck, self-worth, or opportunities.",
    "Trapped in Darkness": "Symbolizes fear of the unknown or feeling lost.",
    "Mirror": "Symbolizes self-reflection, identity, or hidden truths.",
    "Walking on Water": "Represents control over emotions and confidence.",
    "Mountain": "Symbolizes achieving goals or overcoming obstacles.",
    "Bridge": "Indicates a transition or major life change.",
    "Storm": "Reflects emotional turmoil or upcoming difficulties.",
    "Bright Light": "A sign of enlightenment, new ideas, or hope.",
    "Running Out of Time": "Symbolizes pressure or missed opportunities.",
    "Mysterious Key": "Indicates unlocking new possibilities or hidden knowledge.",
    "Staircase": "Symbolizes progress, growth, or spiritual ascent.",
    "Lonely Island": "Represents isolation, self-reflection, or independence.",
    "Fire": "Represents passion, transformation, or destruction.",
    "Holding a Baby": "Symbolizes new beginnings, responsibilities, or creativity.",
    "Talking to the Dead": "Could indicate longing, closure, or subconscious messages.",
    "Out-of-Control Car": "Indicates loss of control in life or reckless decisions.",
    "Unknown Door": "Represents new opportunities, secrets, or unexplored paths.",
}

# Dropdown for dream selection
selected_dream = st.selectbox("🔮 Choose a common dream:", ["Type your own"] + dream_suggestions)

# Input from the user (Auto-fills if a predefined dream is selected)
dream_text = st.text_area("💬 Describe your dream:", value=(selected_dream if selected_dream != "Type your own" else ""))

# Dream Interpretation Function
def interpret_dream(dream_text):
    """AI-powered dream interpretation function."""
    try:
        # Sentiment analysis
        sentiment = sentiment_model(dream_text)[0]
        sentiment_label = sentiment["label"].upper()
        sentiment_score = sia.polarity_scores(dream_text)["compound"]

        # AI-generated dream interpretation
        ai_interpretation = dream_model(
            dream_text, max_length=100, num_return_sequences=1, return_full_text=False
        )[0]["generated_text"]

        # Match dream symbols
        keywords = nltk.word_tokenize(dream_text.lower())
        matched_symbols = [dream_symbols[word.title()] for word in keywords if word.title() in dream_symbols]

        # Construct final interpretation
        final_interpretation = f"### 🌙 **Dream Analysis:**  \n{ai_interpretation}  \n"
        final_interpretation += f"📊 **Sentiment:**  \n{sentiment_label} (Score: {sentiment_score})  \n"
        if matched_symbols:
            final_interpretation += "🔮 **Symbolic Meaning:**  \n" + "\n".join(f"- {s}" for s in matched_symbols)

        return final_interpretation

    except Exception as e:
        return f"🚨 Error: {e}"

# Submit button
if st.button("✨ Interpret My Dream ✨"):
    if dream_text:
        result = interpret_dream(dream_text)
        st.markdown(result, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a dream description.")
