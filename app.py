from flask import Flask, render_template, request
import pandas as pd
from sklearn.externals import joblib

app = Flask(__name__)

# Step 1: Load the machine learning model and label encoder
model_and_encoder = joblib.load('ipl_winner_prediction_model.pkl')

# Extract the model and label encoder from the loaded dictionary
model = model_and_encoder['model']
label_encoder = model_and_encoder['label_encoder']

# Step 2: Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Step 3: Preprocess the user input and make predictions
    city = request.form.get('city')
    venue = request.form.get('Venue')
    team1 = request.form.get('team1')
    team2 = request.form.get('team2')
    toss_decision = request.form.get('toss-decision')
    toss_winner = request.form.get('toss-winner')

    # Create a dictionary with the user input data
    input_data = {
        'city': [city],
        'venue': [venue],
        'team1': [team1],
        'team2': [team2],
        'toss_decision': [toss_decision],
        'toss_winner': [toss_winner]
    }

    # Convert the user input data to a DataFrame
    input_df = pd.DataFrame(input_data)

    # Use the label encoder to transform categorical variables to numerical format
    categorical_columns = ['city', 'venue', 'team1', 'team2', 'toss_winner', 'toss_decision']
    for col in categorical_columns:
        input_df[col] = label_encoder.transform(input_df[col])

    # Step 4: Make predictions using the loaded model
    winner_prediction = model.predict(input_df)
    predicted_winner = label_encoder.inverse_transform(winner_prediction)[0]

    # Step 5: Render the prediction result on the result page
    return render_template('result.html', winner=predicted_winner)

if __name__ == '__main__':
    app.run(debug=True)
