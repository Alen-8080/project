import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pcloud import PyCloud

# Set your pCloud credentials
PCLOUD_USERNAME = "mailjacksparrw14@gmail.com"
PCLOUD_PASSWORD = "password@123"

# Set the folder to monitor for new videos
FOLDER_TO_MONITOR = "/home/raspberry/Desktop/upload_video"

# Set your pCloud upload folder ID
PCLOUD_UPLOAD_FOLDER_ID = "21387003800"

class VideoHandler(FileSystemEventHandler):
    def __init__(self, client):
        self.client = client

    def on_created(self, event):
        if not event.is_directory:
            filename = event.src_path
            if filename.endswith(".jpg"):  # Adjust the file extension as per your video format
                print(f"New video detected: {filename}")
                self.upload_to_pcloud(filename)

    def upload_to_pcloud(self, filename):
        api = PyCloud(PCLOUD_USERNAME, PCLOUD_PASSWORD)
        with open(filename, 'rb') as file_to_upload:
            if os.path.exists(filename):
                # File exists, proceed with upload
                with open(filename, 'rb') as file_to_upload:
                    # The 'files' parameter is a dictionary where the key is 'file' and the value is a tuple of (filename, file object)
                    files = {'file': (os.path.basename(filename), file_to_upload)}
                    # The 'data' parameter is a dictionary that contains other form data, in this case, 'folderid'
                    data = {'folderid': PCLOUD_UPLOAD_FOLDER_ID}
                    upload_result = api.uploadfile(files=files, data=data)
                    if upload_result["result"]:
                        print(f"Uploaded {filename} to pCloud")
                    else:
                        print(f"Failed to upload {filename} to pCloud")
            else:
                print(f"File {filename} not found.")
        api.logout()  # Don't forget to log out when done




if __name__ == "__main__":
    client = PyCloud(PCLOUD_USERNAME, PCLOUD_PASSWORD)
    observer = Observer()
    observer.schedule(VideoHandler(client), path=FOLDER_TO_MONITOR, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
