import os
import json
import shutil
import subprocess
from tqdm import tqdm

def extract_segments(json_file, output_directory):
    with open(json_file, 'r') as f:
        data = f.read()
        segments = {}
        line1 = data.split('\n')[0]
    if line1: 
        segments = json.loads(line1)    
    
    video_name = os.path.splitext(os.path.basename(json_file))[0]
    for idx, (segment_id, segment_info) in tqdm(enumerate(segments.items()), desc=f'Processing {video_name}'):
        start_time = segment_info["startTime"]
        end_time = segment_info["endTime"]
        output_file = os.path.join(output_directory,f'{video_name}_v_{video_name}_{segment_id}.mp4')# convert this 
        input_file = os.path.join(os.path.dirname(json_file), f'{video_name}.mp4')
        ffmpeg_command = f'ffmpeg -i {input_file} -ss {start_time} -to {end_time} -c:v libx264 -c:a aac -strict experimental -tune fastdecode -threads 1 -y {output_file}'
        subprocess.run(ffmpeg_command, shell=True, capture_output=True)

def main():
    json_files_directory = "/home/slasher/Downloads/numerated_videos/"
    all_video_directory = os.path.join(json_files_directory, "all_video")
    special_files = [ "20.json"]
    for filename in os.listdir(json_files_directory):
        if filename.endswith(".json") and filename in special_files:
            json_file = os.path.join(json_files_directory, filename)
            video_name = os.path.splitext(filename)[0]
            output_directory = os.path.join(all_video_directory, video_name)
            os.makedirs(output_directory, exist_ok=True)  # Create output directory if it doesn't exist
            extract_segments(json_file, output_directory)

if __name__ == "__main__":
    main()