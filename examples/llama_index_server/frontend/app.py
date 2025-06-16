# /// script
# dependencies = [
#   "llama-index-server==0.1.19",
# ]
# ///
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from llama_index.server.chat_ui import copy_bundled_chat_ui

if not os.path.exists(".ui"):
    copy_bundled_chat_ui(target_path=".ui")

ui_app = FastAPI(title="Chat UI")
ui_app.mount(
    "/",
    StaticFiles(directory=".ui", html=True),
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(ui_app, port=3001)
