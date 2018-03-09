FROM google/cloud-sdk:190.0.1-slim

RUN cd && mkdir .pip && echo "[global]\nindex-url=http://ftp.daumkakao.com/pypi/simple\ntrusted-host=ftp.daumkakao.com" > ./.pip/pip.conf
RUN sed -i 's/archive.ubuntu.com/ftp.daumkakao.com/g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y python3 python3-pip

WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install google-cloud-dataflow
RUN pip3 install tensorflow==1.4.0
RUN pip3 install tensorflow-transform
RUN pip3 install pillow protobuf
RUN apt-get install -y python-snappy

# https://tensorflow.blog/2017/05/12/tf-%EC%84%B1%EB%8A%A5-%ED%8C%81-winograd-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EC%84%A4%EC%A0%95/
ENV TF_ENABLE_WINOGRAD_NONFUSED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

RUN pip3 install sklearn scipy pandas
RUN pip3 install -U apache-beam[gcp]

# for preventing error (import apache_beam)
RUN pip3 install six==1.10.0

RUN pip3 install numpy==1.12.1
RUN pip3 install click
RUN pip3 install soyspacing

RUN apt-get update && apt-get install -y unzip wget
RUN wget https://github.com/facebookresearch/fastText/archive/master.zip && \
      unzip master.zip && rm master.zip && mv fastText-master fastText && \
      cd fastText && make
RUN cd fastText && pip3 install .

ENV PATH /fastText:$PATH
