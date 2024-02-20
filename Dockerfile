FROM ubuntu:latest
LABEL authors="ibzcl"

ENTRYPOINT ["top", "-b"]