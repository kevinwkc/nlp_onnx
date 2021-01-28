FROM python:3.7-slim
LABEL app=sentiment
MAINTAINER "Kevin Wong"

COPY artifact .

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential=12.6 \
         git \
         libprotoc-dev=3.6.1.3-2 \
         protobuf-compiler=3.6.1.3-2 \
         python-pip=18.1-5 \
         ca-certificates && \
    rm -rf /var/lib/apt/lists/* && pip install -r req.txt
# && pip install -r req_ner.txt
#         vim=2:8.1.0875-5 \
#
WORKDIR /data

VOLUME /data
EXPOSE 80

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["./api.py"]