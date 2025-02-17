# Vosk CLI

A small cli program to demo live transcription using Vosk

## How to run

The project is setup with a pyproject.toml file, so the only thing you need to run it would be to either manually set up a virtual environment or run the following command:
```sh
uv run vosk-cli
```
This will by default download and run an `en-us` model, but you can specify other models using the `-m` parameter.
