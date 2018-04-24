FROM google/cloud-sdk:198.0.0-slim

RUN cd && mkdir .pip && echo "[global]\nindex-url=http://ftp.daumkakao.com/pypi/simple\ntrusted-host=ftp.daumkakao.com" > ./.pip/pip.conf
RUN sed -i 's/archive.ubuntu.com/ftp.daumkakao.com/g' /etc/apt/sources.list

WORKDIR /app
RUN python -m pip install --upgrade pip
RUN pip install google-cloud-dataflow
RUN pip install tensorflow==1.6.0
RUN pip install tensorflow-transform
RUN pip install pillow protobuf
RUN pip install tensor2tensor
RUN apt-get install -y python-snappy

# https://tensorflow.blog/2017/05/12/tf-%EC%84%B1%EB%8A%A5-%ED%8C%81-winograd-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EC%84%A4%EC%A0%95/
ENV TF_ENABLE_WINOGRAD_NONFUSED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

RUN pip install sklearn scipy pandas
RUN pip install -U apache-beam[gcp]

# for preventing error (import apache_beam)
RUN pip install six==1.10.0

RUN pip install numpy==1.13.3
RUN pip install click
