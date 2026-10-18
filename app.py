import os
from flask import Flask, render_template, request, redirect, session, url_for
from utils import load_model, predict_risk

app = Flask(__name__)
# In production, set app.secret_key from an environment variable
app.secret_key = os.getenv("SECRET_KEY", "secret123")

USERNAME = os.getenv("APP_USERNAME", "admin")
PASSWORD = os.getenv("APP_PASSWORD", "1234")

# Lazy-load the trained model bundle safely
# (a dict with the pipeline, tuned decision threshold, model name and feature list)
MODEL_PATH = "model.pkl"
bundle = None


def get_bundle():
    global bundle
    if bundle is None:
        bundle = load_model(MODEL_PATH)
    return bundle


# Numeric form fields: form name -> (model feature, min, max, required)
NUMERIC_FIELDS = {
    "age":        ("age",     18,  120, True),
    "bmi":        ("BMI",     10,   80, True),
    "pir":        ("PIR",      0,   10, False),
    "sbp":        ("SBP",     60,  260, False),
    "dbp":        ("DBP",     30,  150, False),
    "glucose":    ("glucose", 40,  600, False),
    "hdl":        ("HDL",     10,  150, False),
    "tc":         ("TC",      50,  500, False),
    "bun":        ("BUN",      2,  150, False),
    "ast":        ("AST",      5,  500, False),
    "alt":        ("ALT",      5,  500, False),
    "creatinine": (None,     0.2,   15, False),     # converted to eGFR below
}
# Choice fields (blank = unknown, filled in by the model's imputation)
INT_CHOICES = {"sex": ("sex", {1, 2}, True),
               "ethnicity": ("ethnicity", {1, 2, 3, 4, 6, 7}, False),
               "education": ("education", {1, 2, 3, 4, 5}, False),
               "marital": ("marital", {1, 2, 3, 4, 5, 6}, False),
               "hypertension": ("hypertension", {0, 1}, False),
               "diabetes": ("diabetes", {0, 1}, False),
               "dyslipidemia": ("dyslipidemia", {0, 1}, False)}
TEXT_CHOICES = {"smoking": {"current", "former", "never"},
                "drinking": {"current", "former", "never"},
                "activity": {"mild", "moderate", "vigorous"}}


def parse_form(form):
    """Validate the form and return the feature dict for the model. Raises ValueError with a message."""
    values = {}
    for name, (feature, lo, hi, required) in NUMERIC_FIELDS.items():
        raw = form.get(name, "").strip()
        if raw == "":
            if required:
                raise ValueError(f"Please enter {name.upper() if len(name) <= 3 else name}.")
            continue
        try:
            number = float(raw)
        except ValueError:
            raise ValueError(f"'{raw}' is not a valid number for {name}.")
        if not lo <= number <= hi:
            raise ValueError(f"{name} must be between {lo} and {hi}.")
        if feature:
            values[feature] = number
        else:                                  # creatinine
            values["_creatinine"] = number

    for name, (feature, allowed, required) in INT_CHOICES.items():
        raw = form.get(name, "").strip()
        if raw == "":
            if required:
                raise ValueError(f"Please select {name}.")
            continue
        try:
            number = int(raw)
        except ValueError:
            raise ValueError(f"Invalid value for {name}.")
        if number not in allowed:
            raise ValueError(f"Invalid value for {name}.")
        values[feature] = number

    for name, allowed in TEXT_CHOICES.items():
        raw = form.get(name, "").strip().lower()
        if raw == "":
            continue
        if raw not in allowed:
            raise ValueError(f"Invalid value for {name}.")
        values[name] = raw

    # eGFR from creatinine with the MDRD formula used in the base paper
    creatinine = values.pop("_creatinine", None)
    if creatinine is not None:
        egfr = 186 * creatinine ** -1.154 * values["age"] ** -0.203
        if values["sex"] == 2:
            egfr *= 0.742
        values["eGFR"] = egfr
    return values


# ---------------- LOGIN ----------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username_input = request.form.get("username", "").strip()
        password_input = request.form.get("password", "").strip()

        if username_input == USERNAME and password_input == PASSWORD:
            session["user"] = USERNAME
            return redirect(url_for("home"))

        return render_template("login.html", error="Invalid credentials")

    # If user is already logged in, redirect directly to home
    if "user" in session:
        return redirect(url_for("home"))

    return render_template("login.html")


# ---------------- HOME (FORM ONLY) ----------------
@app.route("/home", methods=["GET"])
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


# ---------------- ANALYZE (PROCESS + REDIRECT) ----------------
@app.route("/analyze", methods=["POST"])
def analyze():
    if "user" not in session:
        return redirect(url_for("login"))

    try:
        values = parse_form(request.form)
    except ValueError as e:
        return render_template("index.html", error=str(e), form=request.form)

    # ML inference: probability from the leakage-free pipeline + tuned decision threshold
    try:
        result = predict_risk(values, bundle=get_bundle())
        session["score"] = round(result["probability"] * 100)
        session["flagged"] = bool(result["high_risk"])
        session["model_name"] = get_bundle()["model_name"]
        session["n_inputs"] = len(values)
        session["error"] = None
    except Exception as e:
        print(f"[Model Error] Failed to run prediction: {e}")
        session["score"] = None
        session["flagged"] = None
        session["model_name"] = None
        session["n_inputs"] = len(values)
        session["error"] = "The prediction model could not be run. Please train the model first."

    return redirect(url_for("dashboard"))


# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    # Redirect to home if user visits /dashboard directly without completing the form
    if "n_inputs" not in session:
        return redirect(url_for("home"))

    return render_template(
        "dashboard.html",
        score=session.get("score"),
        flagged=session.get("flagged"),
        model_name=session.get("model_name"),
        n_inputs=session.get("n_inputs"),
        error=session.get("error"),
    )


# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)