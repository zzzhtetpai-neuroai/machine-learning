import urllib.request
import tarfile

# Download the dataset
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filename = "aclImdb_v1.tar.gz"
urllib.request.urlretrieve(url, filename)

# Extract the dataset
with tarfile.open(filename, "r:gz") as tar:
    tar.extractall()

print("Dataset extracted successfully!")
