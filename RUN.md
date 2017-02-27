# RUN

## Build image
```
docker build -t conv-pose . 
```

## Download data
```
cd testing
./get_models.sh
```

## Run docker container + program
```
./run.sh
cd /conv-pose
python run.py
```
