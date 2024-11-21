
```
docker build -t synthetic_data_generator .
```

```
docker run -v $(pwd)/raw_data:/app/raw_data synthetic_data_generator
```
