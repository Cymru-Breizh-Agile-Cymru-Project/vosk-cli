#!/usr/bin/env python3

# prerequisites: as described in https://alphacephei.com/vosk/install and also python module `sounddevice` (simply run command `pip install sounddevice`)
# Example usage using Dutch (nl) recognition model: `python test_microphone.py -m nl`
# For more help run: `python test_microphone.py -h`

import argparse
import json
import queue
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import sounddevice as sd
from huggingface_hub import HfFileSystem, hf_hub_download, repo_exists
from rich import print
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from vosk import KaldiRecognizer, Model

# Since the the Panel isn't able to deal with overflow we just add an arbitrary cap
# at the number of sentences that can be printed at the same time
MAX_SENTENCES = 30


def main():
    q = queue.Queue()

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    args = get_args()

    try:
        # Fetch sample rate from device if it doesn't exist
        if args.samplerate is None:
            device_info = sd.query_devices(args.device, "input")
            # soundfile expects an int, sounddevice provides a float:
            args.samplerate = int(device_info["default_samplerate"])
            print(f"Using a sample rate of {args.samplerate}")

        # Select model
        if args.model is None:
            model = Model(lang="en-us")
        elif Path(args.model).exists():
            model = Model(args.model)
        elif repo_exists(args.model):
            model = load_model_from_huggingface(args.model)
        else:
            model = Model(lang=args.model)

        # Create base layout
        layout = make_layout()
        layout["header"].update(Header())
        layout["footer"].update(Footer(args))
        layout["input"].update(Panel("", title="Live input"))

        with sd.RawInputStream(
            samplerate=args.samplerate,
            blocksize=8000,
            device=args.device,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            rec = KaldiRecognizer(model, args.samplerate)

            completed_sentences = []
            with Live(layout, refresh_per_second=10, screen=True):
                layout["log"].update(TextLog(completed_sentences))

                # Processing loop
                while True:
                    data = q.get()

                    # Processes when a sentence has been completed
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        sentence = result["text"].strip()
                        if sentence:
                            completed_sentences.append(
                                f"[green][{datetime.now().time().isoformat(timespec='seconds')}]:[reset] {sentence}"
                            )
                            layout["log"].update(TextLog(completed_sentences))
                    # Process partial sentence
                    else:
                        result = json.loads(rec.PartialResult())
                        layout["input"].update(
                            Panel(result["partial"], title="Live input")
                        )

    except KeyboardInterrupt:
        print("\nDone")
        exit(0)


def load_model_from_huggingface(model_id: str) -> Model:
    if ":" in model_id:
        model, file_id = model_id.split(":", maxsplit=2)
    else:
        model = model_id
        fs = HfFileSystem()
        tars = fs.glob(f"{model_id}/*.tar.gz")
        if len(tars) == 1:
            print(f"Resolved model tar file to be {tars[0]}")
            file_id = tars[0].removeprefix(model_id).removeprefix("/")
        if len(tars) == 0:
            raise FileNotFoundError(
                f"Could not locate a tar.gz in the repo {model_id}. Use a colon after the model id to specify one"
            )
        if len(tars) > 1:
            ValueError(
                f"Found more than one tar.gz file in the repo {model_id}. Use a colon after the model id to specify one. Found {', '.join(tars)}"
            )

    model_str_path = Path(hf_hub_download(model, file_id))
    extracted_path = model_str_path.parent / model_str_path.stem.split(".")[0]
    print(f"Extracting model to {extracted_path}")
    with tarfile.open(model_str_path) as tar:
        tar.extractall(extracted_path)

    # If there is only one folder in the extracted folder, enter it
    files = [*extracted_path.glob("*")]
    if len(files) == 1 and files[0].is_dir():
        extracted_path = files[0]

    return Model(str(extracted_path))


def make_layout() -> Layout:
    """Define the layout."""
    layout = Layout(name="root")

    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="log"),
        Layout(name="input", size=4),
    )

    return layout


class Header:
    """Display header with clock."""

    def __rich__(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="right", ratio=1)
        grid.add_row(
            "Prifysgol Bangor a ffrindiau",
            "[bold]Vosk Live Demo[reset]",
            "[green]" + datetime.now().ctime().replace(":", "[blink]:[/]"),
        )
        return Panel(grid)


@dataclass
class Footer:
    args: argparse.Namespace

    def __rich__(self) -> Panel:
        components = [
            f"Model: {self.args.model if self.args.model else 'en-us'}",
            f"Sample rate: {self.args.samplerate:,}",
            f"Block size: {8000:,}",
        ]
        return Panel(" | ".join(components), title="Parameters")


@dataclass
class TextLog:
    sentences: list[str]

    def __rich__(self) -> Panel:
        return Panel(
            "\n".join(self.sentences[-MAX_SENTENCES:]),
            title="Sentence log",
        )


def get_args() -> argparse.Namespace:
    def int_or_str(text):
        """Helper function for argument parsing."""
        try:
            return int(text)
        except ValueError:
            return text

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-l",
        "--list-devices",
        action="store_true",
        help="show list of audio devices and exit",
    )
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser],
    )

    parser.add_argument(
        "-d", "--device", type=int_or_str, help="input device (numeric ID or substring)"
    )
    parser.add_argument("-r", "--samplerate", type=int, help="sampling rate")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="language model; e.g. en-us, fr, nl; default is en-us",
    )
    args = parser.parse_args(remaining)
    return args
