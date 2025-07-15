import random
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# updated chatbot with more greetings

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")
print("Model loaded successfully")

# === 1. Prepare sample data ===
train_data = [
    ("Hello", "greet"),
    ("Hi there", "greet"),
    ("Hey", "greet"),
    ("Good morning", "greet"),
    ("Book a flight to Paris", "book_flight"),
    ("I want to fly to London", "book_flight"),
    ("Can you book a ticket to New York?", "book_flight"),
    ("Goodbye", "goodbye"),
    ("Bye", "goodbye"),
    ("See you later", "goodbye")
]

# Split into texts and labels
texts, labels = zip(*train_data)

# === 2. Vectorize the text ===
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# === 3. Train the classifier ===
model = MultinomialNB()
model.fit(X, labels)

# === 4. Responses for each intent ===
responses = {
    "greet": ["Hello!", "Hi there!", "Hey! How can I help you?"],
    "book_flight": ["Sure, where and when would you like to fly?", "I can help with that. What destination and date?"],
    "goodbye": ["Bye! Have a nice day.", "Goodbye!", "See you later!"]
}

# === 5. Function to extract entities ===
def extract_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append((ent.label_, ent.text))
    return entities

# === 6. Chat loop ===
print("🤖 Chatbot is ready! (type 'exit' to quit)")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break
    
    # Predict intent
    X_input = vectorizer.transform([user_input])
    predicted_intent = model.predict(X_input)[0]

    # Generate response
    bot_response = random.choice(responses.get(predicted_intent, ["I'm not sure how to respond to that."]))

    # Print response
    print(f"Bot: {bot_response}")
    
    # If booking, extract entities
    if predicted_intent == "book_flight":
        entities = extract_entities(user_input)
        if entities:
            for label, value in entities:
                print(f"  ↳ Detected {label}: {value}")
        else:
            print("  ↳ Could you please specify the destination and date?")
            