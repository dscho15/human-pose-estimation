import pytube
import argparse
import uuid

if __name__ == "__main__":
    
    args = argparse.ArgumentParser(description="Download a video from YouTube")
    args.add_argument("--url", type=str, default="https://www.youtube.com/watch?v=R3KU9xBvnmU", help="URL of the video to download")
    args.add_argument("--output", type=str, default="./videoes", help="Output directory")
    args = args.parse_args()
    
    yt = pytube.YouTube(args.url)
    yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first().download(args.output, filename=uuid.uuid4().hex + ".mp4")
    
    print("Downloaded video")