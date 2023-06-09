FROM python:3.10-slim-buster

RUN apt-get update && apt-get install -y wget

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN wget https://drive.google.com/file/d/1YzGtJiW-4CSj8O7WAX6T7aNnc7P0wZRN/view?usp=sharing

COPY . .

# Set a non-root user
USER 10014

CMD [ "python", "./main.py" ]
