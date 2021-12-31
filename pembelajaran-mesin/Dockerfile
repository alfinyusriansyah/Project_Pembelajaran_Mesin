FROM python:3.8

WORKDIR /app

EXPOSE 8501

COPY . .

RUN pip install numpy
RUN pip install opencv-python
RUN pip install streamlit
RUN pip install tensorflow
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

CMD ["streamlit","run","./app.py"]