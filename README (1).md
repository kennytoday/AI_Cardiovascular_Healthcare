#   AI cardio Health Risk Checker

This is a streamlit app that estimates a person's 10-year cardiovascular disease risk using the Framingham risk equations. It also includes a Random Forest model trained on the UCI Heart Disease dataset, so you can compare the results from both approaches.

**Note:** This project is for learning and demonstration purposes only. It is not a medical device and should not be used to make medical decisions.

## Features

- **Personalized recommendations:** Provides suggestions based on the user's input, including blood pressure, cholesterol, smoking status, and diabetes.
- **Risk comparison:** Displays the estimated risk alongside an approximate average for people in the same age and sex group.
- **Lifestyle score:** Gives a score from 0 to 100 based on selected health and lifestyle risk factors.
- **Smoking what-if option:** Allows smokers to see how their estimated risk changes when smoking is switched off in the calculation.
- **Input warnings:** Flags values that may be unusually high or need attention.
- **Downloadable results:** Lets users save their results and recommendations as a text file.
- **Combined results:** Runs both the Framingham calculation and the Random Forest model and displays their results together.
- **Streamlit interface:** Includes a simple layout for entering health information and viewing the results.

## Project Structure

```text
AI-Cardio-Health-Risk-Checker/
├── app.py              # Main Streamlit application
├── framingham.py       # Framingham calculations and helper functions
├── train_model.py      # Trains the Random Forest model
├── models/             # Stores the trained model
└── data/               # Heart disease dataset used for training
```

## Getting Started

### 1. Install the required packages

```bash
pip install streamlit pandas scikit-learn joblib plotly
```

### 2. Prepare the model

Make sure the trained model is available in the `models/` directory. If it hasn't been created yet, run the training script:

```bash
python train_model.py
```

You may need to place the dataset in the `data/` directory before training.

### 3. Run the application

```bash
streamlit run app.py
```

### 4. Open the app

Streamlit will display a local URL in the terminal. It is usually:

`http://localhost:8501`

Open the link in your browser to use the application.

## Available Modes

Mode Description 
|---|---|
| **Basic** | Estimates 10-year cardiovascular risk using the Framingham equations. Requires inputs such as age, cholesterol, and blood pressure. |
| **Advanced** | Uses a Random Forest model trained on the UCI Heart Disease dataset. Requires additional clinical inputs. |
| **Both** | Displays the results from both methods for comparison. |

## Disclaimer

This application is intended for educational purposes and experimentation with cardiovascular risk estimation and machine learning.

-  it is not a certified medical device.
- it does not diagnose medical conditions or recommend treatment.
- its estimates may be inaccurate and should not be interpreted as a medical assessment.
- consult a qualified healthcare professional about any health concerns.
- Do  not use the results to make medical decisions.

## Credits

- Framingham risk equations: D'Agostino et al., *Circulation* (2008).
- Machine learning model: Random Forest trained using the UCI Heart Disease (Cleveland) dataset.
