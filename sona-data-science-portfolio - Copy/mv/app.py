from flask import Flask, render_template, request
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import re
app = Flask(__name__)

# Load the saved NB model and tfdfVectorizer
model = pickle.load(open('mnb_model1.pkl', 'rb'))
cv = pickle.load(open('tfdfvectorizer.pkl', 'rb'))

# Define the same text processing functions you used for training
def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned, '', text)
def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem

def to_lower(text):
    return text.lower()

def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

def stem_txt(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])

def process_text(text):
    f1 = clean(text)
    f2 = is_special(f1)
    f3 = to_lower(f2)
    f4 = rem_stopwords(f3)
    return stem_txt(f4)

# Define a dictionary to store movie reviews and ratings
movie_reviews = {}

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result.html', methods=['POST'])
def result():
    # Extract data from the form submission
    movie_name = request.form['movie_name']
    review = request.form['review']

    # Process the user input
    processed_text = process_text(review)
    transformed_text = cv.transform([processed_text])

    # Predict the sentiment
    prediction = model.predict(transformed_text)

    # Assign a rating based on sentiment
    rating = 5 if prediction[0] == 1 else 2

    # Check if the movie is already in the dictionary
    if movie_name in movie_reviews:
        movie_reviews[movie_name]['reviews'].append({'review': review, 'sentiment': prediction[0], 'rating': rating})
    else:
        movie_reviews[movie_name] = {'reviews': [{'review': review, 'sentiment': prediction[0], 'rating': rating}]}

    # Render the result page with necessary data
    return render_template('result.html', movie_name=movie_name, review=review, sentiment=prediction[0], rating=rating)

@app.route('/aggregate_rating/<movie_name>')
def aggregate_rating(movie_name):
    if movie_name in movie_reviews:
        reviews = movie_reviews[movie_name]['reviews']
        total_ratings = sum(review['rating'] for review in reviews)
        aggregate_rating = total_ratings / len(reviews) if len(reviews) > 0 else 0
        return render_template('aggregate_rating.html', movie_name=movie_name, aggregate_rating=aggregate_rating)
    else:
        return f'Movie {movie_name} not found in reviews.'

if __name__ == '__main__':
    app.run(debug=True)