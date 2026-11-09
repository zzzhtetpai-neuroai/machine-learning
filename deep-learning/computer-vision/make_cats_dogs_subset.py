import shutil
import os
from pathlib import Path

original_dir=Path("data/train")
new_dir=Path("data/cats_and_dogs_small")




def make_subset(subset,start_index,end_index):
    for category in ("cat","dog"):
        dir=(new_dir/subset/category)
        Path.mkdir(dir,parents=True,exist_ok=True)
        fnames=[f"{category}.{i}.jpg" for i in range(start_index,end_index)]
        for fname in fnames:
            shutil.copyfile(src=original_dir/fname,dst=dir/fname)

make_subset("train",0,1000)
make_subset("validation",1000,1500)
make_subset("test",1500,2000)
