# Pixolid SAM 3 Studio

Segment images using **Text Prompts** or **Interactive Points** powered by [SAM 3 (Ultralytics)](https://docs.ultralytics.com/models/sam-3/). Created by [Zvonko Vugreshek](https://zvonkovugreshek.com) & [Pixolid UG](https://pixolid.de).

## Features

- **Text Prompt** — Describe what to segment; finds all instances of the concept
- **Interactive Point** — Click on the image to segment specific objects
- **Individual Downloads** — Download each segment separately as mask or cutout
- Auto-downloads model weights from HuggingFace on first run

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, set **Main file path** to `app.py`
4. Go to **Settings → Secrets** and add:
   ```
   HF_TOKEN = "hf_your_token_here"
   ```
5. Deploy!

### Prerequisites

- A [HuggingFace](https://huggingface.co/settings/tokens) access token
- Approved access to the [SAM 3 model](https://huggingface.co/facebook/sam3)

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set your HuggingFace token via environment variable:
```bash
export HF_TOKEN="hf_your_token_here"
```
Or create `.streamlit/secrets.toml`:
```toml
HF_TOKEN = "hf_your_token_here"
```

## License

SAM 3 model weights are subject to Meta's license. See [facebook/sam3](https://huggingface.co/facebook/sam3).
