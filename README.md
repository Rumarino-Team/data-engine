# Data Engine

A complete data annotation and tracking system with both Python (FastAPI) backend and C++ interface. Supports video object segmentation using SAM 2 and point tracking using CoTracker.

Use `git clone --recursive https://github.com/Rumarino-Team/data-engine.git` when cloning the repo

## Install Dependencies

1. Create Python virtual environment using venv or Anaconda (mainly written and tested on Python 3.12.12)
2. If you didn't use `--recursive` when doing git clone, then run `git submodule update --init --recursive` to make sure `sam2` is downloaded
3. Run `pip install -e .` while in `sam2` directory
4. Run `pip install -e ".[notebooks]"` while in `sam2` directory
5. Return to the data engine directory and run `pip install -r requirements.txt`

## Using the Backend
1. source the venv or Anaconda (depending on your shell)
2. Run `fastapi dev backend/api.py` to use the API manually with hot reload, Run `fastapi run backend/api.py` to use the static API

Add the `--host xxx.xxx.xxx.xxx` and/or `--port xxxx` args to change its ip and port

## Testing the Python Backend
1. Run `python3 backend/tests/tester.py` to test the API automatically and see results under `backend/tests/`

## Angular Web Frontend
Mainly for developing the frontend, for production its recommended to use the Tauri Desktop env instead

1. Change into `frontend-ng/data-engine`
2. Install dependencies with `bun install` or `npm install`
3. Start the frontend with `bun run start` or `npm run start`
4. Keep the Python backend running separately

The frontend defaults to `http://127.0.0.1:8000` for backend requests and also exposes a compact API URL override in the top bar.

## Tauri Desktop Frontend

1. Change into `frontend-ng/data-engine`
2. Install JavaScript dependencies with `bun install` or `npm install`
3. Install the Rust toolchain and Tauri prerequisites for your OS
4. Start the Python backend separately
5. Run `bun run tauri:dev` or `npm run tauri:dev:npm` for the desktop dev workflow
6. Run `bun run tauri:build` or `npm run tauri:build:npm` to produce a packaged desktop build

The Tauri app only wraps the Angular frontend. It does not bundle or launch the Python backend.

## Testing via C++
Currently this part is behind the python api's backend implementation so it isnt recommended to use

1. Install `nlohmann_json` and `curl` dev packages via your package manager
2. Change into `/cpp-backend-bindings`
2. Run `mkdir build`
3. Run `cmake build` to generate CMake build files
4. Run `cmake --build build` to compile project
5. Run `./build/src/BackendInterfaceTests` while the venv is sourced to test backend code


## Structure
TBA
