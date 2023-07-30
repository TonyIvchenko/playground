# Playground

Contains multiple projects.

## Installation

```bash
make setup
```

## Usage

```bash
make test
make build-service
make start-service
make train-hurricane-model
make build-hurricane-service
make start-hurricane-service
make smoke-hurricane-service
make train-wildfire-model
make build-wildfire-service
make start-wildfire-service
make smoke-wildfire-service
make bundle-hf-hurricane-space
make bundle-hf-wildfire-space
```

## Hurricane Service

- Service code: `src/hurricane_service/`
- Notebook: `notebooks/hurricane_eda_and_baseline.ipynb`
- Training script: `scripts/train_hurricane_intensity_model.py`
- API endpoints: `/health`, `/predict`, `/service-metadata`
- Built-in Gradio UI: `/ui`
- Implementation plan: `docs/hurricane-service-implementation-plan.md`

## Wildfire Service

- Service code: `src/wildfire_service/`
- Notebook: `notebooks/wildfire_eda_and_baseline.ipynb`
- Training script: `scripts/train_wildfire_ignition_model.py`
- API endpoints: `/health`, `/predict`, `/service-metadata`
- Built-in Gradio UI: `/ui`
- Implementation plan: `docs/wildfire-service-implementation-plan.md`

## Hugging Face Spaces

Build standalone Docker Space bundles:

- `make bundle-hf-hurricane-space`
- `make bundle-hf-wildfire-space`

Generated output:

- `dist/hf_spaces/hurricane/`
- `dist/hf_spaces/wildfire/`

Each bundle includes `Dockerfile`, `README.md` Space metadata, service source, sample data, and training script.

## Contributing

To report issues or contribute code, please see CONTRIBUTING.md.

## License

This project is licensed under the MIT License. See LICENSE.txt for more information.

## Contact

For questions or comments about the project, contact maintainers at toxa.ivchenko@gmail.com.
