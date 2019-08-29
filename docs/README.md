# Docker for API

You can build and run the docker using the following process:

Cloning
```console
git clone https://github.com/jqueguiner/selfie2anime.git selfie2anime 
```

Building Docker
```console
cd selfie2anime && docker build -t selfie2anime -f Dockerfile .
```

Running Docker
```console
echo "http://$(curl ifconfig.io):5000" && docker run -p 5000:5000 -d REPO_NAME
```

Calling the API
```console
curl -X POST "http://MY_SUPER_API_IP:5000/process" -H "accept: image/jpg" -H "Content-Type: application/json" -d '{"url":"https://img.chefdentreprise.com/Img/BREVE/2017/10/321988/Octave-Klaba-OVH-elu-entrepreneur-annee--LE.jpg"}' --output anime_selfie.jpg
```
