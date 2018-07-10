from PIL import Image
import numpy as np

filename="julia_0.25_0.30"

julia=np.fromfile(filename+".bin",dtype=np.uint8)
julia.shape=[16*16,16*16,3]

im = Image.fromarray(julia)
im.save(filename+".jpg")






