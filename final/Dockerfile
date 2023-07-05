FROM python:3.10.6
WORKDIR /app
COPY ./source/requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY ./source /app
CMD ["bash", "script.sh"]