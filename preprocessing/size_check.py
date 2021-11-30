import PIL 
import os 
from PIL import Image

f = r'/Users/nickhyland/Desktop/abstract' 
for file in os.listdir(f):
    if not file.startswith('.'):
        f_img = f+"/"+file 
        img = Image.open(f_img)
        assert img.size == (128,128)
    