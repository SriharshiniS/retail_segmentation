from flask import Flask, render_template, request, redirect, url_for, session
from datetime import datetime
from flask_session import Session

app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Store user info and per-user data
users = {}  # key: username, value: {"email": ..., "todo": [], "goals": [], "notes": []}

@app.route("/")
def home():
    if "username" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("register"))

@app.route("/register", methods=["GET", "POST"])
def register():
    message = None
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        if username and email:
            if username not in users:
                users[username] = {"email": email, "todo": [], "goals": [], "notes": []}
                session["username"] = username
                return redirect(url_for("dashboard"))
            else:
                message = "Username already exists!"
    return render_template("register.html", message=message)

@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("register"))
    username = session["username"]
    user_data = users[username]
    hour = datetime.now().hour
    greeting = "Good Morning" if hour < 12 else "Good Afternoon" if hour < 18 else "Good Evening"
    return render_template(
        "dashboard.html",
        greeting=greeting,
        username=username,
        todo=user_data["todo"],
        goals=user_data["goals"],
        notes=user_data["notes"]
    )

@app.route("/todo", methods=["POST"])
def add_todo():
    if "username" in session:
        task = request.form.get("task")
        if task:
            users[session["username"]]["todo"].append(task)
    return redirect(url_for("dashboard"))

@app.route("/goals", methods=["POST"])
def add_goal():
    if "username" in session:
        goal = request.form.get("goal")
        if goal:
            users[session["username"]]["goals"].append(goal)
    return redirect(url_for("dashboard"))

@app.route("/notes", methods=["POST"])
def add_note():
    if "username" in session:
        note = request.form.get("note")
        if note:
            users[session["username"]]["notes"].append(note)
    return redirect(url_for("dashboard"))

@app.route("/contact", methods=["GET", "POST"])
def contact():
    message = None
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        msg = request.form.get("message")
        if name and email and msg:
            message = "Thank you for contacting us!"
    return render_template("contact.html", message=message)

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("register"))

if __name__ == "__main__":
    app.run(debug=True)
