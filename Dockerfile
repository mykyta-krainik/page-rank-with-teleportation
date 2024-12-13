ARG PYTHON_VERSION=3.12.7-bookworm

FROM python:${PYTHON_VERSION}

RUN apt-get update && apt-get install -y curl default-jdk nodejs npm build-essential

RUN curl -O https://dlcdn.apache.org/spark/spark-3.5.3/spark-3.5.3-bin-hadoop3.tgz && \
    tar -xvzf spark-3.5.3-bin-hadoop3.tgz && \
    mv spark-3.5.3-bin-hadoop3 /opt/spark && \
    rm spark-3.5.3-bin-hadoop3.tgz

RUN curl -O https://dlcdn.apache.org/hadoop/common/hadoop-3.4.0/hadoop-3.4.0.tar.gz && \
    tar -xvzf hadoop-3.4.0.tar.gz && \
    mv hadoop-3.4.0 /opt/hadoop && \
    rm hadoop-3.4.0.tar.gz

RUN pip install pyspark

ENV SPARK_HOME=/opt/spark
ENV HADOOP_HOME=/opt/hadoop
ENV HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
ENV LD_LIBRARY_PATH=$HADOOP_HOME/lib/native
ENV PATH="$SPARK_HOME/bin:$HADOOP_HOME/bin:$PATH"
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

RUN npm install -g nodemon

WORKDIR /pagerank
COPY . /pagerank

EXPOSE 8080

CMD ["nodemon", "--exec", "python", "./main.py", "--ext", "py"]
