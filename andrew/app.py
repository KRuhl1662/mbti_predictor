import flask
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

@app.route('/')
def home():

    return None


if __name__ == "__main__":
    app.run(debug=True)