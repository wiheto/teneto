# Set the base image
FROM continuumio/anaconda3

# Dockerfile author / maintainer
LABEL author='William H Thompson <hedley@startmail.com>'

# Update
RUN apt-get update && apt-get install -y \
  git \
  libgl1-mesa-glx \
  build-essential \
  python-dev \
  libgomp1 \
  libfontconfig1 \
  libxrender1 \ 
  python-matplotlib \
  python-qt4 
  

ENV DISPLAY :0
RUN pip install pip==19.0.1 
RUN pip install git+https://github.com/wiheto/teneto
RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc
ENTRYPOINT /bin/sh
