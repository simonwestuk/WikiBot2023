from flask import Flask, render_template, request, jsonify
import requests
import urllib.request
import re
import nltk
import bs4 as bsoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

nltk.download('punkt')

url = "https://www.mayoclinic.org/diseases-conditions/hay-fever/symptoms-causes/syc-20373039"
get = requests.get(url)

if get.status_code == 200:
    data = urllib.request.urlopen(url).read()
    text = bsoup.BeautifulSoup(data, "html.parser").find_all('p')
    corpus = ''
    
    for t in text:
        corpus += t.text.lower()
        
    corpus = re.sub("[[].*[]]", "", corpus)
    corpus = re.sub(r'\s+', ' ', corpus)
    sentences = nltk.sent_tokenize(corpus)
else:
    print("Unable to fetch data from the URL.")

def get_response(user_question):
    sentences.append(user_question)
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(sentences)
    similarity = cosine_similarity(vectors[-1], vectors)
    answer = sentences[similarity.argsort()[0][-2]]
    sentences.pop()
    return answer.capitalize()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response_api():
    user_question = request.form["user_question"]
    response = get_response(user_question)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
