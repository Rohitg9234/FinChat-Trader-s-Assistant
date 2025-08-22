import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from video import list_channel_videos
from extract_sub import get_subtitle_whisper
from embed import EmbeddingStore

if __name__ == "__main__":
    URL= input("Enter URL : ")
    Limit= int(input("Enter Limit of number of videos: "))
    videos = list_channel_videos(URL, limit=Limit)
    store = EmbeddingStore()
    for v in videos:
        print(v["title"],"-This video is being operated")
        transcript = get_subtitle_whisper(v["url"],model_name="base", language="en")
        store.add_text(transcript)
        print(v["title"],"-This video is added to the database")
