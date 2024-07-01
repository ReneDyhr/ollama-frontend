FROM python:3.12

WORKDIR /usr/src/app

RUN apt update && apt install -y build-essential curl libc6 libc6-dev

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5050
ENV FLASK_APP=server.py
CMD ["flask", "run", "--host", "0.0.0.0", "--port", "5050"]