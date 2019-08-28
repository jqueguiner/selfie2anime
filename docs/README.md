# Docker for API

You can build and run the docker using the following process:

Cloning
```console
git clone selfie2anime 
```

Building Docker
```console
cd selfie2anime && docker build -t REPO_NAME -f Dockerfile .
```

Running Docker
```console
echo "http://$(curl ifconfig.io):5000" && docker run -p 5000:5000 -d REPO_NAME
```

Calling the API
```console
CALL
```
