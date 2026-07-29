Student Score Prediction

A Flask frontend for a student exam performance predictor. Provides a homepage and a form to submit student details for a maths score prediction using serialized model artifacts.

Tech stack
- Python 3.8+
- Flask
- scikit-learn, pandas
- HTML/CSS (Jinja2 templates)

Access
https://student-score-prediction-5way.onrender.com/

Project layout
- app.py - Flask application and routes
- templates/ - Jinja2 templates (index.html, home.html)
- static/ - CSS and assets
- src/pipeline/ - prediction pipeline and helpers
- artifacts/ - location for model and preprocessor
