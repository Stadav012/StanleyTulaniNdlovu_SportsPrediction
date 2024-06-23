import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle as pkl
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from scipy import stats
import base64

# Load pre-trained Bagging Regressor model
with open('BaggingRegressor_SelectedFeatures.pkl', 'rb') as f:
    model = pkl.load(f)

# Define selected features
selected_features = ['mentality_composure', 'potential', 'wage_eur', 'movement_reactions', 'value_eur']

def preprocess(data):
    # Select relevant columns from the data
    relevant_columns = selected_features.copy()

    # Impute missing values in numeric data
    imputer = SimpleImputer(strategy='median')
    data_imputed = pd.DataFrame(imputer.fit_transform(data[relevant_columns]), columns=relevant_columns)

    return data_imputed

def predict_rating(data, overall_rating=None):
    if overall_rating is not None and isinstance(overall_rating, (pd.Series, np.ndarray)) and len(overall_rating) > 1:
        # Scale the data for multiple samples
        X_scaled = StandardScaler().fit_transform(preprocess(data))
    else:
        # For single sample or no overall rating provided, skip scaling
        X_scaled = preprocess(data).values

    # Predict using Bagging Regressor model
    y_pred = model.predict(X_scaled)

    # Round off predicted values to the nearest whole number
    y_pred_rounded = np.round(y_pred)

    # Calculate confidence interval using scipy
    if overall_rating is not None:
        if isinstance(overall_rating, (pd.Series, np.ndarray)) and len(overall_rating) > 1:
            y_cv_pred = cross_val_predict(model, X_scaled, overall_rating, cv=5)
            mse = mean_squared_error(overall_rating, y_cv_pred)
            std_err = np.sqrt(mse)
            conf_interval = stats.norm.interval(0.95, loc=y_pred_rounded, scale=std_err)
        else:
            conf_interval = (y_pred_rounded, y_pred_rounded)  # No interval if only one sample
    else:
        conf_interval = (y_pred_rounded, y_pred_rounded)  # Default to point estimate if overall_rating is None

    # Create DataFrame with predictions and confidence intervals
    predictions_df = pd.DataFrame({
        'short_name': data['short_name'],  # Assuming 'short_name' is a column in data
        'Predicted Overall Rating': y_pred_rounded,
        'Confidence Interval (Lower)': conf_interval[0],
        'Confidence Interval (Upper)': conf_interval[1]
    })

    # Set index of predictions_df to match data.index
    predictions_df.index = data.index

    return predictions_df

def main():
    st.title('FIFA Player Rating Prediction')
    st.subheader('Input your player data manually or upload a CSV file:')

    # Option to input data manually
    st.sidebar.header("Manual Input")
    short_name = st.sidebar.text_input('Short Name (Player Name)')
    mentality_composure = st.sidebar.number_input('Mentality Composure', min_value=0.0, max_value=100.0)
    potential = st.sidebar.number_input('Potential', min_value=0.0, max_value=100.0)
    wage_eur = st.sidebar.number_input('Wage (EUR)', min_value=0.0)
    movement_reactions = st.sidebar.number_input('Movement Reactions', min_value=0.0, max_value=100.0)
    value_eur = st.sidebar.number_input('Value (EUR)', min_value=0.0)
    overall_rating_manual = st.sidebar.number_input('Overall Rating (Manual Input)', min_value=0.0, max_value=100.0)

    # Option to upload a CSV file
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            # Load CSV data
            data = pd.read_csv(uploaded_file)

            # Check if 'overall' column exists in the data
            if 'overall' in data.columns:
                overall_rating_csv = data['overall']
            else:
                overall_rating_csv = None

            # Predict using Bagging Regressor model
            predictions_df = predict_rating(data, overall_rating_csv)

            # Display predictions
            st.subheader('Predictions:')
            st.write(predictions_df)

            # Option to download predictions as CSV
            csv_download_link = create_download_link(predictions_df, "predicted_ratings.csv", "Download Predictions CSV")
            st.markdown(csv_download_link, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")

    if st.sidebar.button('Predict'):
        # Collect input data into a DataFrame
        input_data = pd.DataFrame({
            'short_name': [short_name],
            'mentality_composure': [mentality_composure],
            'potential': [potential],
            'wage_eur': [wage_eur],
            'movement_reactions': [movement_reactions],
            'value_eur': [value_eur]
        })

        # Predict using Bagging Regressor model
        predictions_df = predict_rating(input_data, overall_rating_manual)

        # Display prediction
        st.subheader('Prediction:')
        st.write(f"Player Name: {short_name}")
        st.write(f"Predicted Overall Rating: {predictions_df.iloc[0, 1]:.0f}")  # Display rounded predicted rating
        st.write(f"Confidence Interval: {predictions_df.iloc[0, 2]:.0f} to {predictions_df.iloc[0, 3]:.0f}")  # Display confidence interval

        # Option to download prediction as CSV
        csv_download_link = create_download_link(predictions_df, "predicted_rating.csv", "Download Prediction CSV")
        st.markdown(csv_download_link, unsafe_allow_html=True)

def create_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64 encoding
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

if __name__ == '__main__':
    main()
