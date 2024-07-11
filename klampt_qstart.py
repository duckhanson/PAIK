import math
from klampt import *
from klampt.math import so3
from klampt.model.trajectory import RobotTrajectory
import time

data_path = "/home/luca/paik/data/visualization_resources/klampt_data"

#some Klamp't code
w = WorldModel()
# data/visualization_resources/klampt_data/tx90scenario0.xml
w.readFile(f"{data_path}/tx90scenario0.xml")
r = w.robot(0)
q0 = r.getConfig()
r.randomizeConfig()
qrand = r.getConfig()
r.setConfig(q0)

#add a "world" item to the scene manager
vis.add("world",w)
#show qrand as a ghost configuration in transparent red
vis.add("qrand",qrand,color=(1,0,0,0.5))
#show a Trajectory between q0 and qrand
# vis.add("path_to_qrand",RobotTrajectory(r,[0,1],[q0,qrand]))

#To control interaction / animation, launch the loop via one of the following:

#some more Klamp't code to load a geometry
mug_path = "/home/luca/paik/data/visualization_resources/objects/mug.obj"

# load a mug and attach it to the end effector of the robot
mug = Geometry3D()
mug.loadFile(mug_path)
T = r.link(7).getTransform()
mug.setCurrentTransform(*T)
vis.add("mug",mug)

#OpenGL on Linux / windows
vis.show()              #open the window
t0 = time.time()
cnt = 0
cnt_up = True
path = [q0,qrand]

while vis.shown():
    if cnt_up:
        cnt += 1
    else:
        cnt -= 1
    
    if cnt == 100:
        cnt_up = False
    elif cnt == 0:
        cnt_up = True
    
    print(f"cnt: {cnt}")    
    
    # a smooth trajectory between q0 and qrand
    q = r.interpolate(path[0],path[1],cnt/100.0)
    r.setConfig(q)
    
    T = r.link(6).getTransform()
    # ratate the mug around the y-axis of the end effector with 90 degrees
    T = (so3.from_axis_angle(([0,1,0],math.radians(270))),T[1])
    mug.setCurrentTransform(*T)
    # print(mug.getCurrentTransform())
    #do something, e.g. the following
    # if time.time() > 5:
    #     vis.setColor("qrand",0,1,1,0.5)  #sets qrand to show in cyan after 5 seconds
    time.sleep(0.1)    #loop is called ~100x times per second
vis.kill()              #safe cleanup

#Mac OpenGL workaround: launch the vis loop and window in single-threaded mode
#vis.loop()


