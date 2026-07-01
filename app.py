from model import hybrid_translate
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import json
import hashlib

app = Flask(__name__)
app.secret_key = "tourism_translation_secret_key"

USERS_FILE = "users.json"


def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as file:
        return json.load(file)


def save_users(users):
    with open(USERS_FILE, "w") as file:
        json.dump(users, file)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# temporary fallback translator
def translate_text(text):
    fallback = {
        "hello": "bonjour",
        "how are you": "comment allez-vous ?",
        "good morning": "bonjour",
        "thank you": "merci",
        "welcome": "bienvenue",
        "hotel": "hôtel",
        "airport": "aéroport",
        "tourism": "tourisme",
        "travel": "voyage",
        "beach": "plage",
        "food": "nourriture"
    }

    cleaned = text.lower().strip().replace(".", "").replace("?", "")
    return fallback.get(cleaned, "Translation unavailable. Please try a simpler sentence.")


@app.route("/")
def home():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", username=session["username"])


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        users = load_users()
        username = request.form["username"]
        password = hash_password(request.form["password"])

        if username in users and users[username] == password:
            session["username"] = username
            return redirect(url_for("home"))

        return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        users = load_users()
        username = request.form["username"]
        password = hash_password(request.form["password"])

        if username in users:
            return render_template("signup.html", error="Username already exists")

        users[username] = password
        save_users(users)

        return redirect(url_for("login"))

    return render_template("signup.html")

@app.route("/translate", methods=["POST"])
def translate():
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    text = data.get("text", "")

    translation, source = hybrid_translate(text)

    return jsonify({
        "translation": translation,
        "source": source
    })


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
   app.run(debug=True)