FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY airbnb_streamlit_dashboard.py .
COPY Airbnb_Open_Data.csv .
COPY rf_model.joblib .
COPY feature_names.joblib .

# Use environment variable for port, default to 8501
ENV STREAMLIT_PORT=8501
EXPOSE $STREAMLIT_PORT

ENV STREAMLIT_SERVER_PORT=$STREAMLIT_PORT
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

ENTRYPOINT ["sh", "-c", "streamlit run airbnb_streamlit_dashboard.py --server.port $STREAMLIT_PORT --server.address 0.0.0.0"]