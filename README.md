# Playground

Contains multiple minimal services.

## Dependency Model

1. Root `requirements.txt` contains only general environment dependencies.
2. Each service under `src/<service>/requirements.txt` contains its own dependencies.
3. Root `environment.yml` references root requirements plus every service requirements file.
4. When adding a new service, add its folder under `src/` with `main.py`, `Dockerfile`, and `requirements.txt`, then add that requirements path to `environment.yml`.

## Installation

```bash
make setup
```

## Usage

```bash
make test
make build-hurricane-service
make start-hurricane-service
make smoke-hurricane-service
make build-wildfire-service
make start-wildfire-service
make smoke-wildfire-service
```

## Local Run Without Docker

```bash
cd src
PYTHONPATH=. API_PORT=8000 python3 -m hurricane_service.main
PYTHONPATH=. API_PORT=8010 python3 -m wildfire_service.main
```

## Service Structure

Every service is kept minimal:
- `main.py`
- `requirements.txt`
- `Dockerfile`

## Contributing

To report issues or contribute code, please see CONTRIBUTING.md.

## License

This project is licensed under the MIT License. See LICENSE.txt for more information.
