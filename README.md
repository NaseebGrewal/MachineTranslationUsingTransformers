# MachineTranslationUsingTransformers

A project demonstrating machine translation using transformer models. This repository includes multiple scripts for building and fine-tuning transformer-based models for multilingual translation tasks, with a focus on English to French translation. The project explores the basics of transformer architectures and provides hands-on examples for translation between various languages.

## Project Structure

- `.gitignore`: Git ignore file to exclude unnecessary files from the repository.
- `.pre-commit-config.yaml`: Configuration file for pre-commit hooks to maintain code quality.
- `LICENSE`: The license for the repository.
- `README.md`: This file.
- `building_transformer_model.py`: Script to build and train a transformer model for machine translation.
- `en2fr_translation.py`: A specific script for translating English to French using a transformer model.
- `hello_world.py`: Basic script for initializing and testing a transformer model.
- `multilanguage_translation.py`: Script for handling translation between multiple languages.
- `requirements-ci.txt`: Dependencies required for Continuous Integration (CI).
- `requirements-dev.txt`: Development dependencies.
- `requirements.txt`: Main dependencies for the project.
- `test.ipynb`: Jupyter notebook for testing and validating the translation models.

## Requirements

This project uses Python 3.x and several libraries related to machine learning, natural language processing, and transformers. The dependencies are listed in the `requirements.txt` and `requirements-dev.txt` files.

To install the necessary dependencies:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/MachineTranslationUsingTransformers.git
    cd MachineTranslationUsingTransformers
    ```

2. Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. For development purposes, you can also install the development dependencies:
    ```bash
    pip install -r requirements-dev.txt
    ```

5. To use pre-commit hooks (optional but recommended):
    ```bash
    pip install pre-commit
    pre-commit install
    ```

## Scripts

### `building_transformer_model.py`
This script demonstrates how to build and train a transformer model for machine translation tasks. It uses common libraries such as `transformers` from Hugging Face.

### `en2fr_translation.py`
A specific implementation to translate text from English to French using the transformer model.

### `hello_world.py`
A minimalistic script for testing if the transformer model is working properly.

### `multilanguage_translation.py`
This script extends the functionality to handle translation tasks across multiple languages, allowing more flexible usage of the transformer models.

### `test.ipynb`
A Jupyter notebook for experimenting with the translation model and testing its performance in a hands-on manner.

## Pre-commit Hooks

The project uses pre-commit hooks to ensure that the code follows a consistent format. This is configured in the `.pre-commit-config.yaml` file.

To install the hooks:
```bash
pre-commit install
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Feel free to fork the repository, open issues, or submit pull requests. Contributions are always welcome!
