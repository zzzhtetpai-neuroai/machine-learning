import urllib.request
import tarfile
import sys

# URLs for the Oxford Pets dataset
images_url = "http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
annotations_url = "http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

def download_with_progress(url, filename):
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = downloaded / total_size * 100 if total_size > 0 else 0
        sys.stdout.write(f"\rDownloading {filename}: {percent:.2f}%")
        sys.stdout.flush()
    urllib.request.urlretrieve(url, filename, reporthook)
    print(f"\n{filename} download complete!")

def extract_tar(filename):
    print(f"Extracting {filename}...")
    with tarfile.open(filename) as tar:
        tar.extractall()
    print(f"{filename} extraction complete!")

# Download with progress
download_with_progress(images_url, "images.tar.gz")
download_with_progress(annotations_url, "annotations.tar.gz")

# Extract files
extract_tar("images.tar.gz")
extract_tar("annotations.tar.gz")

print("All downloads and extractions complete!")
