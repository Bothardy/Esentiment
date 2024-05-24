import streamlit as st
import joblib
import re
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import firebase_admin
from firebase_admin import credentials, firestore
import nltk
import streamlit.components.v1 as components

# Ensure NLTK resources are available
nltk.download('stopwords')
nltk.download('wordnet')

# Check if Firebase app is already initialized
if not firebase_admin._apps:
    # Initialize Firebase app (replace with your Firebase project credentials)
    cred = credentials.Certificate("authenticatorapp-10fc8-firebase-adminsdk-m7efw-bb6797ccd9.json")
    firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()

# Load the trained model and TF-IDF vectorizer
model = joblib.load('model.pkl')
tfidf = joblib.load('vector.pkl')

# Assume you have saved the model accuracy
model_accuracy = 0.85  # For example purposes, replace with the actual accuracy

# Define text preprocessing functions
def cleanstr(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# Function to generate HTML content dynamically
def generate_html_content(reviews):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta http-equiv="X-UA-Compatible" content="IE=edge" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Customer Review</title>
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900&display=swap">
        <style>
          * {
            margin: 0;
            padding: 0;
            font-family: 'Poppins', sans-serif;
            box-sizing: border-box;
          }
          a {
            text-decoration: none;
          }
          #header {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            height: auto;
            background-color: #f8f9fa;
          }
          #header img {
            width: 100%;
            height: auto;
          }
          #testimonials {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            width: 100%;
          }
          .testimonial-heading {
            letter-spacing: 1px;
            margin: 30px 0;
            padding: 10px 20px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
          }
          .testimonial-heading span {
            font-size: 1.3rem;
            color: #252525;
            margin-bottom: 10px;
            letter-spacing: 2px;
            text-transform: uppercase;
          }
          .testimonial-box-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            width: 90%;
            padding: 20px;
          }
          .testimonial-box {
            width: 100%;
            box-shadow: 2px 2px 30px rgba(0, 0, 0, 0.1);
            background-color: #ffffff;
            padding: 20px;
            margin: 15px;
            cursor: pointer;
            transition: transform 0.3s ease;
          }
          .profile-img {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            overflow: hidden;
            margin-right: 10px;
          }
          .profile-img img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            object-position: center;
          }
          .profile {
            display: flex;
            align-items: center;
          }
          .name-user {
            display: flex;
            flex-direction: column;
          }
          .name-user strong {
            color: #3d3d3d;
            font-size: 1.1rem;
            letter-spacing: 0.5px;
          }
          .name-user span {
            color: #979797;
            font-size: 0.8rem;
          }
          .reviews {
            color: #f9d71c;
          }
          .box-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
          }
          .client-comment p {
            font-size: 0.9rem;
            color: #4b4b4b;
          }
          .testimonial-box:hover {
            transform: translateY(-10px);
          }
          @media(max-width: 1060px) {
            .testimonial-box {
                padding: 10px;
            }
          }
          @media(max-width: 790px) {
            .testimonial-box {
                width: 100%;
            }
            .testimonial-heading h4 {
                font-size: 1.4rem;
            }
          }
          @media(max-width: 340px) {
            .box-top {
                flex-wrap: wrap;
                margin-bottom: 10px;
            }
            .reviews {
                margin-top: 10px;
            }
          }
          ::selection {
            color: #ffffff;
            background-color: #252525;
          }
        </style>
    </head>
    <body>
        <section id="header">
          <img src="https://www.ultrapc.ma/24601-large_default/msi-pulse-15-b13vgk-i9-13900h-32gb-1tb-ssd-rtx4070-8gb-156-165hz-ips-qhd.jpg" alt="Product Image">
        </section>
        <section id="testimonials">
          <div class="testimonial-heading">
              <span>Crosshair 16 HX MONSTER HUNTER EDITION D14V</span>
              <br>
              <br>
              <h2>What Master DSEF Students say about this product </h2>
          </div>
          <div class="testimonial-box-container">
    """

    for review in reviews:
        review_data = review.to_dict()
        user_name = "Anonymous"
        user_handle = "@anonymous"
        user_img = "https://cdn3.iconfinder.com/data/icons/avatars-15/64/_Ninja-2-512.png"
        review_text = review_data.get("review_text", "No review text")
        predicted_sentiment = "Positive" if review_data.get("predicted_sentiment", 0) == 1 else "Negative"
        review_rating = 5 if predicted_sentiment == "Positive" else 1

        review_box = f"""
            <div class="testimonial-box">
                <div class="box-top">
                    <div class="profile">
                        <div class="profile-img">
                            <img src="{user_img}" />
                        </div>
                        <div class="name-user">
                            <strong>{user_name}</strong>
                            <span>{user_handle}</span>
                        </div>
                    </div>
                    <div class="reviews">
                        {"".join('<i class="fas fa-star"></i>' for _ in range(review_rating))}
                        {"".join('<i class="far fa-star"></i>' for _ in range(5 - review_rating))}
                    </div>
                </div>
                <div class="client-comment">
                    <p>{review_text}</p>
                </div>
            </div>
        """
        html_content += review_box

    html_content += """
          </div>
        </section>
        <script src='https://kit.fontawesome.com/c8e4d183c2.js'></script>
      </body>
    </html>
    """
    return html_content

# Function to update reviews and refresh the HTML content
def update_reviews_and_html():
    reviews = list(db.collection("reviews").stream())
    html_content = generate_html_content(reviews)

    # Calculate the dynamic height based on the number of reviews
    num_reviews = len(reviews)  # Convert reviews to a list to get the count
    height_per_review = 300  # Approximate height for each review box
    base_height = 600  # Base height for the page without reviews
    dynamic_height = base_height + num_reviews * height_per_review

    # Display the updated HTML content
    components.html(html_content, height=dynamic_height)

# Fetch and display the reviews when the app starts
update_reviews_and_html()

# Section for adding new reviews directly
st.title("Share your thoughts")
new_review_text = st.text_area("Review this product : ")

if st.button("Submit Review"):
    if new_review_text:
        # Preprocess the input text
        cleaned_text = cleanstr(new_review_text)
        cleaned_text = remove_stopwords(cleaned_text)
        cleaned_text = lemmatize(cleaned_text)

        # Transform the text using the TF-IDF vectorizer
        text_vector = tfidf.transform([cleaned_text])

        # Predict the sentiment using the loaded model
        prediction = model.predict(text_vector)[0]

        # Determine the number of stars based on sentiment
        review_rating = 5 if prediction == 1 else 1

        # Get the current timestamp
        timestamp = datetime.datetime.now()

        # Save the new review to Firestore
        new_review_data = {
            "review_text": new_review_text,
            "predicted_sentiment": int(prediction),
            "timestamp": timestamp,
            "review_rating": review_rating
        }
        db.collection("reviews").add(new_review_data)
        st.success("Review submitted successfully!")

        # Rerun the app to update the content
        st.experimental_rerun()
    else:
        st.error("Please enter a review text.")
