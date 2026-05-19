#importing required libraries

from flask import Flask, flash, request, redirect, url_for, render_template
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import load_model
from textblob import TextBlob
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from urllib.parse import urlparse, parse_qs
from werkzeug.utils import secure_filename
app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/uploads"
app.secret_key = '1a2b3c4d5e'
# Load the saved vocabulary


# Allowed extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename):
    """Check if the uploaded file is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
# Load Tesseract OCR

# Load the saved vocabulary for text vectorization
with open("vectorizer_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# Rebuild the vectorizer and set its vocabulary
loaded_vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=1800, output_mode='int')
loaded_vectorizer.set_vocabulary(vocab)

# Load the trained model
loaded_model = load_model("youtube_comment_classification_model.h5", compile=False)


# Define toxicity categories
categories = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if username == "admin" and password == "admin":
            flash("Welcome!", "success")
            # go where the user likely wants
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password.", "danger")
            return render_template("Login.html")
    return render_template("Login.html")


@app.route("/logout")
def logout():
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


@app.route('/home')
def home():
    return render_template('index.html')
    # User is not loggedin redirect to login page

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/category')
def category():
    return render_template('category.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/imageupload')
def imageupload():
    return render_template('imageupload.html')



def predict_toxicity(comment):
    """Predict toxicity of extracted text."""
    vectorized_comment = loaded_vectorizer([comment])
    vectorized_comment = tf.convert_to_tensor(vectorized_comment)
    vectorized_comment = tf.reshape(vectorized_comment, (1, 1800))
    
    prediction = loaded_model.predict(vectorized_comment)[0]
    result_dict = {categories[i]: round(float(prediction[i]), 4) for i in range(len(categories))}
    return result_dict


def get_comments(video_id1):
    # empty list for storing reply
    api_service_name = "youtube"
    api_version = "v3"
    developer_key = "AIzaSyAT8qQvy9VtD1xHGYW3Az3X5HzAyzhDrKA"
    youtube = build(api_service_name, api_version, developerKey=developer_key)

    # Fetch the comments for a specific video
    video_id = video_id1
    comments = []
    next_page_token = None
    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            pageToken=next_page_token
        )
        response = request.execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return comments


def extract_video_id(url):
    try:
        # Remove whitespace
        url = url.strip()
        
        # If it's already just a video ID (11 characters, alphanumeric)
        if len(url) == 11 and url.replace('-', '').replace('_', '').isalnum():
            return url
        
        query = urlparse(url)
        
        # Handle youtu.be short URLs
        if query.hostname == 'youtu.be':
            video_id = query.path.lstrip('/')
            if video_id:
                return video_id.split('?')[0]  # Remove any query params
        
        # Handle youtube.com URLs
        elif query.hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
            if query.path == '/watch':
                if 'v' in parse_qs(query.query):
                    return parse_qs(query.query)['v'][0]
            elif query.path.startswith('/embed/'):
                return query.path.split('/embed/')[1].split('?')[0]
            elif query.path.startswith('/v/'):
                return query.path.split('/v/')[1].split('?')[0]
        
        return None  # Invalid URL
    except Exception as e:
        print(f"Error extracting video ID: {e}")
        return None



@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Show the YouTube URL prediction page
    if request.method == "GET":
        return render_template("contact.html")

    # POST: handle YouTube URL, fetch comments, run model, and show detailed stats
    try:
        input_data = request.form.get("url", "").strip()
        if not input_data:
            return render_template("contact.html", error="Please enter a YouTube URL")

        video_id = extract_video_id(input_data)
        if not video_id or video_id == input_data:
            return render_template("contact.html", error="Invalid YouTube URL. Please enter a valid YouTube video URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID)")

        print(f"Extracted video ID: {video_id}")
        
        try:
            comments = get_comments(video_id)
        except Exception as api_error:
            print(f"YouTube API Error: {api_error}")
            return render_template("contact.html", error=f"Error fetching comments: {str(api_error)}. Please check if the video has comments enabled and try again.")
        
        if not comments:
            return render_template("contact.html", error="No comments found for this video. The video may have comments disabled.")
        
        print(f"Found {len(comments)} comments")

        df = pd.DataFrame(comments, columns=["comment"])

        # Threshold and which categories to summarize in the UI
        # Lowered threshold so more categories (Insult, Obscene, Threat, Identity Hate)
        # are counted when they have a moderate score.
        TOXIC_THRESHOLD = 0.30
        # Use all 6 categories, including "Toxic", when aggregating totals
        selected_cats = categories

        results = []
        total_count = 0

        # Count comments that cross the threshold per category (for ALL categories you have)
        cat_counts = {cat: 0 for cat in categories}

        # Count comments that are toxic in ANY of the selected categories
        selected_overall_count = 0

        for comment in df["comment"]:
            # Skip empty comments
            if not isinstance(comment, str) or comment.strip() == "":
                continue

            total_count += 1

            # Vectorize
            vectorized_comment = loaded_vectorizer([comment])
            vectorized_comment = tf.convert_to_tensor(vectorized_comment)
            vectorized_comment = tf.reshape(vectorized_comment, (1, 1800))

            # Toxicity prediction (array of 6 scores in the same order as `categories`)
            prediction = loaded_model.predict(vectorized_comment)[0]
            scores = {categories[i]: round(float(prediction[i]), 2) for i in range(len(categories))}

            # Update per-category counts
            for cat, score in scores.items():
                if score >= TOXIC_THRESHOLD:
                    cat_counts[cat] += 1

            # Update overall (selected categories) count
            toxic_cats_for_comment = [cat for cat in selected_cats if scores[cat] >= TOXIC_THRESHOLD]
            if toxic_cats_for_comment:
                selected_overall_count += 1

            # Build a simple final label for this comment
            final_label = ", ".join(toxic_cats_for_comment) if toxic_cats_for_comment else "Non-toxic"

            # Sentiment analysis
            polarity = TextBlob(comment).sentiment.polarity
            sentiment = 'Positive' if polarity > 0 else ('Negative' if polarity < 0 else 'Neutral')

            results.append({
                "comment": comment,
                "toxicity": scores,
                "sentiment": sentiment,
                "final_label": final_label,
            })

        # Percentages per category
        cat_percentages = {
            cat: round((cat_counts[cat] / total_count) * 100, 1) if total_count > 0 else 0
            for cat in categories
        }

        # Determine the dominant toxic category across all comments (if any)
        dominant_category = None
        dominant_percentage = 0
        if total_count > 0 and any(count > 0 for count in cat_counts.values()):
            dominant_category = max(categories, key=lambda c: cat_counts[c])
            dominant_percentage = cat_percentages.get(dominant_category, 0)

        # Human‑readable final overall result for the whole video
        if total_count == 0:
            final_overall_result = "No comments were available to analyze."
            final_overall_level = "neutral"
        elif selected_overall_count == 0:
            final_overall_result = f"Final Result: Non-toxic \u2013 0 out of {total_count} comments crossed the toxicity threshold."
            final_overall_level = "non_toxic"
        else:
            main_cat_text = f" Most common toxic type: {dominant_category} ({dominant_percentage}% of comments)." if dominant_category else ""
            final_overall_result = (
                f"Final Result: Toxic \u2013 {selected_overall_count} out of {total_count} comments "
                f"crossed the toxicity threshold.{main_cat_text}"
            )
            final_overall_level = "toxic"

        print(f"Processing complete. Total comments: {total_count}, Results: {len(results)}")
        
        return render_template(
            "contact.html",
            results=results if results else [],
            youtube_url=input_data,
            toxic_threshold=TOXIC_THRESHOLD,
            total_count=total_count,
            cat_counts=cat_counts,
            cat_percentages=cat_percentages,
            selected_overall_count=selected_overall_count,
            dominant_category=dominant_category,
            dominant_percentage=dominant_percentage,
            final_overall_result=final_overall_result,
            final_overall_level=final_overall_level,
        )

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print("ERROR in /predict:", e)
        print("Full traceback:", error_trace)
        return render_template("contact.html", error=f"An error occurred: {str(e)}. Please check the console for details.")



@app.route('/upload', methods=['GET', 'POST'])
@app.route("/upload1", methods=["GET", "POST"])
def upload1():
    if request.method == "GET":
        return render_template("SocialMedia.html", comments=[], res=0)

    file = request.files.get("file")

    if not file or file.filename == "":
        return render_template("SocialMedia.html", error="No file uploaded", comments=[], res=0)

    if not file.filename.endswith(".csv"):
        return render_template("SocialMedia.html", error="File must be a .csv", comments=[], res=0)

    data = pd.read_csv(file)

    if "comment" not in data.columns:
        return render_template("SocialMedia.html", error="CSV must have a 'comment' column", comments=[], res=0)

    toxic_comments = []

    for comment in data["comment"]:
        if not isinstance(comment, str) or comment.strip() == "":
            continue

        vectorized_comment = loaded_vectorizer([comment])
        vectorized_comment = tf.convert_to_tensor(vectorized_comment)
        vectorized_comment = tf.reshape(vectorized_comment, (1, 1800))

        prediction = loaded_model.predict(vectorized_comment)[0]
        scores = {categories[i]: round(float(prediction[i]), 2) for i in range(len(categories))}

        if any(score > 0.5 for score in scores.values()):
            toxic_comments.append((comment, scores))

    return render_template("SocialMedia.html", comments=toxic_comments, res=1)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)






    