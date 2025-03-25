# Vosk TUI

A small tui program to demo live transcription using Vosk

## Prerequisites
The microphone library uses `libportaudio2` so you need that installed to run the project. On Ubuntu this can be installed globally by running `sudo apt install libportaudio2`.

## How to run

The project is setup with a pyproject.toml file, so the only thing you need to run it would be to either manually set up a virtual environment or run the following command:
```sh
uv run vosk-tui
```
This will by default download and run an `en-us` model, but you can specify other models using the `-m` parameter.
