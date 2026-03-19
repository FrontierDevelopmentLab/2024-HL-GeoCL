FROM python:3.11-slim

EXPOSE 8080

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN pip3 install -r requirements.txt

CMD streamlit run --server.port 8080 --server.enableCORS false app.py
