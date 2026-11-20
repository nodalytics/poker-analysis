# The 9-max test platform (testplatform/ninemax_platform.py): a tournament against rule bots, for an unattended
# Kestrel bot session. It binds 127.0.0.1 only, so compose runs it in Kestrel's network namespace.
#   docker build -f kestrel/docker/testplatform.Dockerfile -t kestrel-testplatform kestrel
FROM python:3.12-slim
RUN pip install --no-cache-dir pokerkit==0.7.7 && useradd --create-home --uid 10002 platform
COPY testplatform/ninemax_platform.py /app/ninemax_platform.py
USER platform
ENTRYPOINT ["python", "/app/ninemax_platform.py"]
CMD ["--port", "8793"]
