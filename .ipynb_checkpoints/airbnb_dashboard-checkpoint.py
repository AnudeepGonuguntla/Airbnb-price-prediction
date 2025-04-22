import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# Load dataset
df = pd.read_csv('Airbnb_Open_Data.csv')

# Data preprocessing
# Clean price and service fee columns (remove '$' and convert to float)
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
df['service fee'] = df['service fee'].replace('[\$,]', '', regex=True).astype(float)

# Select features and target
features = ['neighbourhood group', 'room type', 'lat', 'long', 'Construction year', 
            'minimum nights', 'number of reviews', 'reviews per month', 
            'review rate number', 'calculated host listings count', 'availability 365']
target = 'price'

# Drop rows where target is missing
df = df.dropna(subset=[target])

# Split data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numeric and categorical columns
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

# Create and train model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
model.fit(X_train, y_train)

# Get feature importance
feature_names = (numeric_features + 
                 model.named_steps['preprocessor']
                 .named_transformers_['cat']
                 .named_steps['onehot']
                 .get_feature_names_out(categorical_features).tolist())
importances = model.named_steps['regressor'].feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Initialize Dash app
app = Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1('Airbnb Price Prediction Dashboard', style={'textAlign': 'center'}),
    
    html.Div([
        html.Label('Select Neighbourhood Group:'),
        dcc.Dropdown(
            id='neighbourhood-dropdown',
            options=[{'label': ng, 'value': ng} for ng in df['neighbourhood group'].unique() if pd.notna(ng)],
            value=None,
            placeholder='All Neighbourhoods',
            style={'width': '50%'}
        ),
    ], style={'margin': '10px'}),
    
    html.Div([
        html.Label('Select Room Type:'),
        dcc.Dropdown(
            id='room-type-dropdown',
            options=[{'label': rt, 'value': rt} for rt in df['room type'].unique() if pd.notna(rt)],
            value=None,
            placeholder='All Room Types',
            style={'width': '50%'}
        ),
    ], style={'margin': '10px'}),
    
    html.Div([
        dcc.Graph(id='feature-importance-plot')
    ], style={'width': '50%', 'display': 'inline-block'}),
    
    html.Div([
        dcc.Graph(id='price-distribution-plot')
    ], style={'width': '50%', 'display': 'inline-block'}),
])

# Callback to update plots
@app.callback(
    [Output('feature-importance-plot', 'figure'),
     Output('price-distribution-plot', 'figure')],
    [Input('neighbourhood-dropdown', 'value'),
     Input('room-type-dropdown', 'value')]
)
def update_plots(neighbourhood, room_type):
    # Filter data based on selections
    filtered_df = df.copy()
    if neighbourhood:
        filtered_df = filtered_df[filtered_df['neighbourhood group'] == neighbourhood]
    if room_type:
        filtered_df = filtered_df[filtered_df['room type'] == room_type]
    
    # Feature importance plot
    fig1 = px.bar(
        feature_importance_df.head(10),
        x='Importance',
        y='Feature',
        title='Top 10 Feature Importance (Random Forest)',
        orientation='h'
    )
    fig1.update_layout(yaxis={'autorange': 'reversed'})
    
    # Price distribution plot
    fig2 = px.histogram(
        filtered_df,
        x='price',
        nbins=50,
        title='Price Distribution',
        labels={'price': 'Price ($)'}
    )
    fig2.update_layout(bargap=0.1)
    
    return fig1, fig2

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)