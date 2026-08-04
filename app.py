from flask import Flask, render_template, request, redirect, url_for, session
from gmail.fetch_emails import get_latest_emails
from predict import predict_email
from datetime import datetime
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# -------------------------------
# Server-side storage for scanned emails
# -------------------------------
scan_state = {
    "gmail_results": [],
    "spam_count": 0,
    "ham_count": 0,
    "total": 0,
    "last_scan": None,
}

def page():
    return render_template(
        "index.html",
        gmail_results=scan_state["gmail_results"],
        spam_count=scan_state["spam_count"],
        ham_count=scan_state["ham_count"],
        total=scan_state["total"],
        last_scan=scan_state["last_scan"],
        prediction_result=session.pop("prediction_result", None),
        confidence_result=session.pop("confidence_result", None),
        subject_input=session.pop("subject_input", ""),
        body_input=session.pop("body_input", "")
    )
@app.route("/")
def index():
    return render_template(
        "index.html",
        gmail_results=scan_state["gmail_results"],
        spam_count=scan_state["spam_count"],
        ham_count=scan_state["ham_count"],
        total=scan_state["total"],
        last_scan=scan_state["last_scan"],
        prediction_result=session.pop("prediction_result", None),
        confidence_result=session.pop("confidence_result", None),
        subject_input=session.pop("subject_input", ""),
        body_input=session.pop("body_input", "")
    )

@app.route("/scan", methods=["POST"])
def scan():
    max_results = int(request.form.get("max_results", 10))
    print("Requested:", max_results)

    emails = get_latest_emails(max_results)
    print("Fetched:", len(emails))

    gmail_results = []
    spam_count = 0
    ham_count = 0

    for email in emails:
        prediction, confidence = predict_email(email["subject"], email["body"])
        label = "SPAM" if prediction == 1 else "HAM"
        if label == "SPAM":
            spam_count += 1
        else:
            ham_count += 1

        gmail_results.append({
            "subject": email["subject"],
            "body": email["body"][:350],
            "prediction": label,
            "confidence": round(confidence, 2) if confidence is not None else None
        })

    scan_state["gmail_results"] = gmail_results
    scan_state["spam_count"] = spam_count
    scan_state["ham_count"] = ham_count
    scan_state["total"] = len(gmail_results)
    scan_state["last_scan"] = datetime.now().strftime("%d %b %Y %I:%M %p")

    return redirect(url_for("index"))

@app.route("/predict", methods=["POST"])
def predict():
    subject = request.form.get("subject", "")
    body = request.form.get("body", "")

    prediction, confidence = predict_email(subject, body)

    session["prediction_result"] = "SPAM" if prediction == 1 else "HAM"
    session["confidence_result"] = round(confidence, 2) if confidence is not None else None
    session["subject_input"] = subject
    session["body_input"] = body

    return redirect(url_for("index"))

@app.route("/clear", methods=["POST"])
def clear():

    scan_state["gmail_results"] = []
    scan_state["spam_count"] = 0
    scan_state["ham_count"] = 0
    scan_state["total"] = 0
    scan_state["last_scan"] = None

    session.clear()

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)