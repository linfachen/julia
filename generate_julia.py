from PIL import Image
import numpy as np
import subprocess
import os
import sys

def generate_julia(x,y,shape=[1024,1536,3]):
    filename="julia_%0.3f_%0.3f" % (x,y)
    executable="julia"
    if sys.platform=="win32":
        executable+=".exe"
    else:
    	executable ="./"+executable    
    cmd=[executable,str(x),str(y)]
    subprocess.check_output(cmd)
    
    julia=np.fromfile(filename+".bin",dtype=np.uint8)
    julia.shape=shape
    im = Image.fromarray(julia)
    im.save(filename+".jpg")
    im.close()
    os.remove(filename+".bin")

if __name__=="__main__":
    generate_julia(-0.8, -0.156)    



