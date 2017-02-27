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
./run_container.sh
python run.py
```
