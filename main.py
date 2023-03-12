import typer
import whisper
import openai
import os
from pytube import YouTube

openai.api_key = os.getenv("GPTKEY")

app = typer.Typer()

@app.command()
def summarize(link: str = typer.Argument(..., help="Youtube link"), model: str = typer.Argument("base.en", help="Whisper model, small, base, medium, large")):

    whis = whisper.load_model(model)
    video = YouTube(link).streams.first().download()
    message = whis.transcribe(video, fp16=False)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": f"summarize {message['text']}"}
        ],
    )
    print(response['choices'][0]['message']['content'])


if __name__ == '__main__':
    app()