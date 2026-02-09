import pandas as pd
import plotly.express as px
import streamlit as st

from model_training import load_or_train_bundle

st.set_page_config(page_title="Airbnb Price Intelligence", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("Airbnb_Open_Data.csv")
    df["price"] = pd.to_numeric(df["price"].replace("[\\$,]", "", regex=True), errors="coerce")
    df = df.dropna(subset=["price"])
    return df


@st.cache_resource
def load_bundle():
    return load_or_train_bundle()


df = load_data()
bundle = load_bundle()
model = bundle["model"]
metrics = bundle["metrics"]
feature_importance_df = bundle["importances"]
defaults = bundle["defaults"]

sample_size = min(10000, len(df))
df_viz = df.sample(n=sample_size, random_state=42)

st.title("Airbnb Price Intelligence Dashboard")
st.caption(
    "Project redefined for accuracy: richer features + log-transformed target + model selection."
)

with st.sidebar:
    st.header("Model Quality")
    st.metric("Best Model", metrics["best_model"])
    st.metric("R²", f"{metrics['r2']:.3f}")
    st.metric("MAE", f"${metrics['mae']:.2f}")
    st.caption(f"Train rows: {metrics['train_rows']:,} | Test rows: {metrics['test_rows']:,}")

st.header("Predict Price")
with st.form("prediction_form"):
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        pred_neighbourhood_group = st.selectbox(
            "Neighbourhood Group", sorted(df["neighbourhood group"].dropna().unique())
        )
        pred_neighbourhood = st.selectbox(
            "Neighbourhood",
            sorted(
                df[df["neighbourhood group"] == pred_neighbourhood_group][
                    "neighbourhood"
                ]
                .dropna()
                .unique()
            ),
        )
        pred_room_type = st.selectbox("Room Type", sorted(df["room type"].dropna().unique()))

    with col_b:
        pred_instant = st.selectbox(
            "Instant Bookable", sorted(df["instant_bookable"].dropna().unique())
        )
        pred_cancel = st.selectbox(
            "Cancellation Policy", sorted(df["cancellation_policy"].dropna().unique())
        )
        pred_min_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, value=2)

    with col_c:
        pred_reviews = st.number_input("Number of Reviews", min_value=0, value=10)
        pred_reviews_month = st.number_input(
            "Reviews per Month", min_value=0.0, value=float(defaults["reviews per month"])
        )
        pred_availability = st.number_input(
            "Availability 365", min_value=0, max_value=365, value=120
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame(
        {
            "neighbourhood group": [pred_neighbourhood_group],
            "neighbourhood": [pred_neighbourhood],
            "room type": [pred_room_type],
            "instant_bookable": [pred_instant],
            "cancellation_policy": [pred_cancel],
            "lat": [defaults["lat"]],
            "long": [defaults["long"]],
            "Construction year": [defaults["Construction year"]],
            "minimum nights": [pred_min_nights],
            "number of reviews": [pred_reviews],
            "reviews per month": [pred_reviews_month],
            "review rate number": [defaults["review rate number"]],
            "calculated host listings count": [defaults["calculated host listings count"]],
            "availability 365": [pred_availability],
        }
    )

    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Price: ${prediction:.2f}")

    benchmark = (
        df_viz[(df_viz["room type"] == pred_room_type)]
        .groupby("room type")["price"]
        .mean()
        .reset_index()
    )
    predicted_row = pd.DataFrame({"room type": ["Predicted"], "price": [prediction]})
    comp = pd.concat([benchmark, predicted_row], ignore_index=True)

    fig_pred = px.bar(
        comp,
        x="room type",
        y="price",
        title=f"Predicted Price vs Room-Type Baseline ({pred_room_type})",
        labels={"price": "Price ($)", "room type": "Room Type"},
    )
    st.plotly_chart(fig_pred, use_container_width=True)

st.sidebar.header("Data Filters")
neighbourhood_filter = st.sidebar.selectbox(
    "Neighbourhood Group",
    options=["All"] + sorted([ng for ng in df_viz["neighbourhood group"].dropna().unique()]),
)
room_filter = st.sidebar.selectbox(
    "Room Type",
    options=["All"] + sorted([rt for rt in df_viz["room type"].dropna().unique()]),
)

filtered_df = df_viz.copy()
if neighbourhood_filter != "All":
    filtered_df = filtered_df[filtered_df["neighbourhood group"] == neighbourhood_filter]
if room_filter != "All":
    filtered_df = filtered_df[filtered_df["room type"] == room_filter]

col1, col2 = st.columns(2)

with col1:
    fig1 = px.bar(
        feature_importance_df.head(12),
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Top Feature Importances ({metrics['best_model']})",
    )
    fig1.update_layout(yaxis={"autorange": "reversed"})
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(
        filtered_df,
        x="price",
        nbins=35,
        title="Observed Price Distribution",
        labels={"price": "Price ($)"},
    )
    fig2.update_layout(bargap=0.1)
    st.plotly_chart(fig2, use_container_width=True)
