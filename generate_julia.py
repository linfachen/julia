from PIL import Image
import numpy as np
import subprocess


def generate_julia(x,y,shape=[512,512,3]):
    filename="julia_%0.3f_%0.3f" % (x,y)
    cmd=["julia.exe",str(x),str(y)]
    subprocess.check_output(cmd)
    
    julia=np.fromfile(filename+".bin",dtype=np.uint8)
    julia.shape=shape
    im = Image.fromarray(julia)
    im.save(filename+".jpg")


if __name__=="__main__":
    generate_julia(-0.3,0.156)    



