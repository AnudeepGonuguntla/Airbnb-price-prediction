FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY airbnb_streamlit_dashboard.py .
COPY Airbnb_Open_Data.csv .
COPY rf_model.joblib .
COPY feature_names.joblib .
EXPOSE 8501
CMD ["streamlit", "run", "airbnb_streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
