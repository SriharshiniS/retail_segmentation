from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Temporary storage (Beginner level – no database)
students = [
    {"name": "Alice", "age": 20, "course": "Computer Science"},
    {"name": "Bob", "age": 21, "course": "Mechanical Engineering"}
]

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/students")
def student_list():
    return render_template("students.html", students=students)


@app.route("/add", methods=["GET", "POST"])
def add_student():

    if request.method == "POST":
        name = request.form.get("name")
        age = request.form.get("age")
        course = request.form.get("course")

        if name and age and course:
            students.append({
                "name": name,
                "age": age,
                "course": course
            })

        return redirect(url_for("student_list"))

    return render_template("add_student.html")


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
