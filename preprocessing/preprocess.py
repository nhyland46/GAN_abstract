import PIL 
import os 
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

f = r'/Users/nickhyland/Desktop/abstract' 
i = 0 
for file in os.listdir(f):
    
    f_img = f+"/"+file 
    try:
        img = Image.open(f_img)
        img = img.resize((128,128)) 
        img = img.convert('RGB')
        img.save(f_img)
        
    except PIL.UnidentifiedImageError:
        print("unidentified image")
