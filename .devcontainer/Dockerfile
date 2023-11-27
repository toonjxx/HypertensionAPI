FROM nvcr.io/nvidia/pytorch:23.10-py3
COPY ./requirements.txt /workspace/requirements.txt
RUN mkdir /python && cd /python && \
    wget https://www.python.org/ftp/python/3.11.6/Python-3.11.6.tgz && \
    tar -zxvf Python-3.11.6.tgz && \
    cd Python-3.11.6 && \
    ls -lhR && \
    ./configure --enable-optimizations && \
    make install && \
    rm -rf /python \
    ln -sf /usr/local/python3/bin/python3 /usr/bin/python3 \
    ln -sf /usr/local/python3/bin/pip3 /usr/bin/pip3 ;

RUN pip3 install -r requirements.txt
