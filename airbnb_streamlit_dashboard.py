import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import plotly.express as px
import joblib
import os
import streamlit as st

# Set page config for better layout
st.set_page_config(page_title="Airbnb Price Prediction new", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Airbnb_Open_Data.csv')
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
    df['service fee'] = df['service fee'].replace('[\$,]', '', regex=True).astype(float)
    return df

df = load_data()

# Data preprocessing
features = ['neighbourhood group', 'room type', 'lat', 'long', 'Construction year', 
            'minimum nights', 'number of reviews', 'reviews per month', 
            'review rate number', 'calculated host listings count', 'availability 365']
target = 'price'

# Drop rows where target is missing
df = df.dropna(subset=[target])

# Sample a subset for visualization
df_viz = df.sample(n=10000, random_state=42)

# Load or train model
model_file = 'rf_model.joblib'
if os.path.exists(model_file):
    model = joblib.load(model_file)
    feature_names = joblib.load('feature_names.joblib')
else:
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = ['lat', 'long', 'Construction year', 'minimum nights', 
                        'number of reviews', 'reviews per month', 'review rate number', 
                        'calculated host listings count', 'availability 365']
    categorical_features = ['neighbourhood group', 'room type']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
    ])
    model.fit(X_train, y_train)

    joblib.dump(model, model_file)
    feature_names = (numeric_features + 
                     model.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_features).tolist())
    joblib.dump(feature_names, 'feature_names.joblib')

# Get feature importance
importances = model.named_steps['regressor'].feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Streamlit app
st.title("Airbnb Price Prediction Dashboard")

# Price prediction tool
st.header("Predict Price")
with st.form("prediction_form"):
    pred_neighbourhood = st.selectbox("Neighbourhood Group", df['neighbourhood group'].unique())
    pred_room_type = st.selectbox("Room Type", df['room type'].unique())
    pred_min_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, value=1)
    pred_reviews = st.number_input("Number of Reviews", min_value=0, value=10)
    pred_availability = st.number_input("Availability 365", min_value=0, max_value=365, value=100)
    
    submitted = st.form_submit_button("Predict")
    if submitted:
        # Use default values for unused features
        input_data = pd.DataFrame({
            'neighbourhood group': [pred_neighbourhood],
            'room type': [pred_room_type],
            'lat': [df_viz['lat'].mean()],
            'long': [df_viz['long'].mean()],
            'Construction year': [df_viz['Construction year'].mean()],
            'minimum nights': [pred_min_nights],
            'number of reviews': [pred_reviews],
            'reviews per month': [df_viz['reviews per month'].mean()],
            'review rate number': [df_viz['review rate number'].mean()],
            'calculated host listings count': [df_viz['calculated host listings count'].mean()],
            'availability 365': [pred_availability]
        })
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Price: ${prediction:.2f}")

        # Visualization: Predicted price vs. average price by room type
        avg_price_room = df_viz.groupby('room type')['price'].mean().reset_index()
        pred_df = pd.DataFrame({
            'room type': [pred_room_type, 'Predicted'],
            'price': [avg_price_room[avg_price_room['room type'] == pred_room_type]['price'].values[0], prediction]
        })
        fig_pred = px.bar(
            pred_df,
            x='room type',
            y='price',
            title=f'Predicted Price vs. Average for {pred_room_type}',
            labels={'price': 'Price ($)', 'room type': 'Room Type'}
        )
        fig_pred.update_layout(height=400)
        st.plotly_chart(fig_pred, use_container_width=True)

# Sidebar for filters
st.sidebar.header("Filters")
neighbourhood = st.sidebar.selectbox(
    "Select Neighbourhood Group",
    options=[None] + [ng for ng in df_viz['neighbourhood group'].unique() if pd.notna(ng)],
    format_func=lambda x: "All Neighbourhoods" if x is None else x
)
room_type = st.sidebar.selectbox(
    "Select Room Type",
    options=[None] + [rt for rt in df_viz['room type'].unique() if pd.notna(rt)],
    format_func=lambda x: "All Room Types" if x is None else x
)

# Filter data
filtered_df = df_viz.copy()
if neighbourhood:
    filtered_df = filtered_df[filtered_df['neighbourhood group'] == neighbourhood]
if room_type:
    filtered_df = filtered_df[filtered_df['room type'] == room_type]

# Layout with columns for visualizations
col1, col2 = st.columns(2)

# Feature importance plot
with col1:
    fig1 = px.bar(
        feature_importance_df.head(10),
        x='Importance',
        y='Feature',
        title='Top 10 Feature Importance (Random Forest)',
        orientation='h'
    )
    fig1.update_layout(yaxis={'autorange': 'reversed'})
    st.plotly_chart(fig1, use_container_width=True)

# Price distribution plot
with col2:
    fig2 = px.histogram(
        filtered_df,
        x='price',
        nbins=30,
        title='Price Distribution',
        labels={'price': 'Price ($)'}
    )
    fig2.update_layout(bargap=0.1)
    st.plotly_chart(fig2, use_container_width=True)
