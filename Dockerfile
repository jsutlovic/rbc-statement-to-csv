FROM python:3.12-bookworm

WORKDIR /app
VOLUME /work

COPY requirements-docker.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD convert_chequing.sh
