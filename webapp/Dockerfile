FROM python:3.10-slim 
WORKDIR /app 
COPY . . 
RUN pip3 install --upgrade pip 
RUN pip3 install -r requirements.txt

EXPOSE 8081

ENTRYPOINT ["streamlit", "run", "api.py", "--server.port=8081", "--server.address=0.0.0.0"]