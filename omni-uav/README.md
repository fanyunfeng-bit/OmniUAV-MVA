# OmniUAV

Python UI with two tabs:

- Multi-UAV camera selection and visualization switching
- Scene point cloud rendering

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Point cloud input

The point cloud tab supports:

- `.xyz` / `.txt` (space-separated X Y Z columns)
- `.npy` (Nx3 NumPy array)

## LLM configuration

The LLM panel uses an OpenAI-compatible API.

Set environment variables before running:

- `LLM_API_KEY`
- `LLM_BASE_URL` (optional, default `https://api.openai.com/v1`)
- `LLM_MODEL` (optional, default `gpt-4o-mini`)

## Demo video inputs

Use the "Select Video Folder" button in the UI to choose a folder that contains:

- `cam01.mp4`, `cam02.mp4`, `cam03.mp4`, `cam04.mp4`

