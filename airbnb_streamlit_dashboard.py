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

# Set page config
st.set_page_config(page_title="Airbnb Price Prediction Dashboard", layout="wide")

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
df_viz = df.sample(n=5000, random_state=42)  # Reduced to 5000 for speed

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

# Cache feature importance
@st.cache_data
def get_feature_importance():
    importances = model.named_steps['regressor'].feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    return feature_importance_df.sort_values(by='Importance', ascending=False)

feature_importance_df = get_feature_importance()

# Streamlit app
st.title("Airbnb Price Prediction Dashboard")

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

# Summary statistics
st.header("Summary Statistics")
st.write(f"Number of Listings: {len(filtered_df)}")
st.write(f"Average Price: ${filtered_df['price'].mean():.2f}")
st.write(f"Median Price: ${filtered_df['price'].median():.2f}")

# Price prediction tool
st.header("Predict Airbnb Price")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        pred_neighbourhood = st.selectbox("Neighbourhood Group", df['neighbourhood group'].unique())
        pred_room_type = st.selectbox("Room Type", df['room type'].unique())
        pred_lat = st.number_input("Latitude", min_value=40.0, max_value=41.0, value=40.7)
        pred_long = st.number_input("Longitude", min_value=-74.5, max_value=-73.0, value=-73.9)
        pred_construction_year = st.number_input("Construction Year", min_value=2000, max_value=2023, value=2010)
    with col2:
        pred_min_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, value=1)
        pred_reviews = st.number_input("Number of Reviews", min_value=0, value=10)
        pred_reviews_per_month = st.number_input("Reviews per Month", min_value=0.0, value=1.0)
        pred_review_rate = st.number_input("Review Rate Number", min_value=1, max_value=5, value=4)
        pred_listings_count = st.number_input("Host Listings Count", min_value=1, value=1)
        pred_availability = st.number_input("Availability 365", min_value=0, max_value=365, value=100)
    
    submitted = st.form_submit_button("Predict Price")
    if submitted:
        input_data = pd.DataFrame({
            'neighbourhood group': [pred_neighbourhood],
            'room type': [pred_room_type],
            'lat': [pred_lat],
            'long': [pred_long],
            'Construction year': [pred_construction_year],
            'minimum nights': [pred_min_nights],
            'number of reviews': [pred_reviews],
            'reviews per month': [pred_reviews_per_month],
            'review rate number': [pred_review_rate],
            'calculated host listings count': [pred_listings_count],
            'availability 365': [pred_availability]
        })
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Price: ${prediction:.2f}")

# Visualizations
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
        nbins=20,  # Reduced bins
        title='Price Distribution',
        labels={'price': 'Price ($)'}
    )
    fig2.update_layout(bargap=0.1)
    st.plotly_chart(fig2, use_container_width=True)
# Map visualization
st.header("Airbnb Listings Map")
if len(filtered_df) == 0:
    st.warning("No valid data available for the map after filtering. Try adjusting the filters.")
elif filtered_df[['lat', 'long']].isna().any().any():
    st.warning("Missing latitude or longitude values in the filtered data. Cannot render map.")
else:
    try:
        # Limit to 1000 points to avoid overloading
        map_df = filtered_df.sample(n=min(1000, len(filtered_df)), random_state=42)
        fig_map = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="long",
            color="price",
            size="number of reviews",
            hover_data=["neighbourhood group", "room type", "price"],
            color_continuous_scale=px.colors.sequential.Plasma,
            title="Airbnb Listings by Price",
            zoom=10
        )
        fig_map.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig_map, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to render map: {str(e)}")
        # Fallback: Scatter plot
        st.subheader("Fallback: Scatter Plot of Listings")
        fig_fallback = px.scatter(
            filtered_df,
            x="long",
            y="lat",
            color="price",
            size="number of reviews",
            hover_data=["neighbourhood group", "room type", "price"],
            title="Airbnb Listings (Scatter Plot)",
            labels={"long": "Longitude", "lat": "Latitude"}
        )
        st.plotly_chart(fig_fallback, use_container_width=True)