# Some tools necessary to convert a video to 1fps format #
import os
import shutil
import cv2


def organize_videos(source_folder):
    # Ensure the source folder exists
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' does not exist.")
        return

    # Get a list of all files in the source folder
    file_list = os.listdir(source_folder)

    # Filter video files (you can add more video file extensions if needed)
    video_extensions = ['.mp4', '.avi', '.mkv']
    video_files = [file for file in file_list if any(file.lower().endswith(ext) for ext in video_extensions)]

    if not video_files:
        print("No video files found in the source folder.")
        return

    # Create a subfolder for each video file and move the video file into it
    for video_file in video_files:
        video_name, _ = os.path.splitext(video_file)
        video_folder = os.path.join(source_folder, video_name)

        # Create the subfolder if it doesn't exist
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)

        # Move the video file into the subfolder
        source_path = os.path.join(source_folder, video_file)
        destination_path = os.path.join(video_folder, video_file)

        shutil.move(source_path, destination_path)
        print(f"Moved '{video_file}' to '{video_folder}'.")

    print("All video files have been organized.")


def extract_frames_from_videos(source_folder):
    # Ensure the source folder exists
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' does not exist.")
        return

    # Iterate over subfolders (assumed to be folders containing videos)
    for subfolder_name in os.listdir(source_folder):
        subfolder_path = os.path.join(source_folder, subfolder_name)

        if not os.path.isdir(subfolder_path):
            continue

        # Iterate over video files in the subfolder
        for video_name in os.listdir(subfolder_path):
            video_path = os.path.join(subfolder_path, video_name)

            # Check if the current item is a file and has a video extension
            if not os.path.isfile(video_path):
                continue

            video_basename, video_extension = os.path.splitext(video_name)

            video_folder = os.path.join(subfolder_path, video_basename)
            os.makedirs(video_folder, exist_ok=True)

            # Open the video using OpenCV
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            frame_index = 0
            while frame_index < frame_count:
                ret, frame = cap.read()

                if not ret:
                    break

                # Sample one frame per second
                if int(frame_index % fps) == 0:
                    frame_filename = f"{video_basename}_frame{frame_index}.jpg"
                    frame_path = os.path.join(video_folder, frame_filename)
                    cv2.imwrite(frame_path, frame)

                frame_index += 1

            cap.release()

            print(f"Extracted frames from '{video_name}' to '{video_folder}'.")

    print("Frame extraction completed.")



if __name__ == '__main__':
    dir_fps_path = '../data/PERCEPTION/avqa-frames-1fps'
    organize_videos(dir_fps_path)
    extract_frames_from_videos(dir_fps_path)
