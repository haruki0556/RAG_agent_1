FROM python:3.12

WORKDIR /app

COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD chainlit run app.py --host 0.0.0.0 --port 8000



