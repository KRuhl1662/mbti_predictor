import flask
import re
import string
import nltk
import pandas as pd
import joblib
import pickle
from flask import Flask, render_template, jsonify, request
from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField, IntegerField, FloatField
from wtforms.validators import DataRequired
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

nltk.download('stopwords')
nltk.download('wordnet')


class PredictorForm(FlaskForm):
    input1 = TextAreaField('Enter sentence here.', validators=[DataRequired()])
    submit = SubmitField('Submit')


with open('mbti_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('vector.pickle', 'rb') as cv:
   vec = pickle.load(cv)

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

app = Flask(__name__)

app.config['SECRET_KEY'] = 'mbti'

@app.route('/homepage', methods=['GET','POST'])
def home():
    
    form = PredictorForm()

    if request.method == 'POST':
        input_one_text = form.input1.data

        predicted_value = Predict(form)

        return render_template('index.html', form = form, predicted=predicted_value)
    else:
        return render_template('index.html', form = form)

@app.route('/visuals')
def visuals():

    return render_template('visuals.html')

@app.route('/contact')
def contacts():

    return render_template('contact.html')

@app.route('/data1')
def data1():
    # use these for json data
    return None

@app.route('/data2')
def data2():
    # use these for json data
    return None

@app.route('/data3')
def data3():
    # use these for json data
    return None

@app.route("/data", methods=['GET', 'POST'])

def Predict(honey):

    input1_df = honey.input1.data


    honey_predict_df = pd.DataFrame({
        'Input 1': [input1_df]
    })

    model_input = honey_predict_df['Input 1']

    posts = []

    stemmer = WordNetLemmatizer()

    for text in range(0, len(model_input)):
        post = re.sub(r'\W', ' ', str(model_input[text]))
        post = re.sub(r'\s+[a-zA-Z]\s+', ' ', post)
        post = re.sub(r'\^[a-zA-Z]\s+', ' ', post) 
        post = re.sub(r'\s+', ' ', post, flags=re.I)
        post = post.lower()
        post = post.split()
        post = [stemmer.lemmatize(word) for word in post]
        post = ' '.join(post)
        posts.append(post)

    x = vec.fit_transform(posts).toarray()

    y_pred = model.predict(x)

    return y_pred

if __name__ == "__main__":
    app.run(debug=True)