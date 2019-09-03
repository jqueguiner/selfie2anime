FROM guignol95/ai_apis:latest

RUN apt-get update -y

RUN apt-get install -y unrar \
		language-pack-en

ADD src /src

WORKDIR /src

RUN pip3 install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python3"]

CMD ["app.py"]
