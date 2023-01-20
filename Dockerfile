FROM python:3.9.6

EXPOSE 8501
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt


#CMD streamlit run app/streamlit/stream.py
ENTRYPOINT ["streamlit", "run", "app/streamlit/stream.py", "--server.port=8501", "--server.address=0.0.0.0"]
