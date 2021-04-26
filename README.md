# ITA Fase 1

To install enviroment, run `conda env create --name ita`
To create the ipykernel, run `python -m ipykernel install --user --name ita --display-name "ita"`


## Autogluon container 

> Files will be written in `ita_fase_1/src/autogluon/out` of the host machine

Building with **docker-compose**

in `ita_fase_1/src/autogluon/`

```
docker compose up -d
```

Building and running with **docker build**

in `ita_fase_1/`

```
docker build -t autogluon:v1 -f src/autogluon/Dockerfile .

docker run --name autogluon_app -v "$(pwd)"/src/autogluon:/app/src/autogluon autogluon:v1
```