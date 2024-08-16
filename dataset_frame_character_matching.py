import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt


# Function to convert time (HH:MM:SS) to seconds and formatted string
def convert_time_to_seconds(time_obj):
    # Extract hours, minutes, and seconds from the time object
    hours = time_obj.hour
    minutes = time_obj.minute
    seconds = time_obj.second

    # Calculate the total number of seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds

    # Format the time string in HH:MM:SS format
    formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"

    # Return the formatted time and total seconds
    return formatted_time, total_seconds


# Function to save frames to corresponding character folders
def save_frames(output_dir, characters, video_file, start_time, end_time,show_frame = False):
    folder_name = '_'.join(characters)
    folder_path = os.path.join(output_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Unable to open video file '{video_file}'")
        return

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frame rate: {frame_rate}")

    # Convert start_time and end_time using the new function
    start_time_formatted, start_time_seconds = convert_time_to_seconds(start_time)
    end_time_formatted, end_time_seconds = convert_time_to_seconds(end_time)
    print(f"start: {start_time_formatted}, end: {end_time_formatted}")

    start_frame = int(start_time_seconds * frame_rate)
    end_frame = int(end_time_seconds * frame_rate)

    print(f"Start frame: {start_frame}, End frame: {end_frame}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to read frame {current_frame}")
            break

        # Show the frame
        if show_frame:
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title(f"Frame {current_frame}")
            plt.axis('off')
            plt.show()

        # Save the frame
        frame_filename = os.path.join(folder_path, f"{folder_name}_frame_{current_frame:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved frame {current_frame} as {frame_filename}")

        current_frame += 1

    cap.release()
