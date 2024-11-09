FROM python: 3.8-alpine
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD streamlit run app.py --server.port 8502