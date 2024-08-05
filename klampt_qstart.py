from jrl.robots import Baxter



robot = Baxter()
qr, pr = robot.sample_joint_angles_and_poses(5, only_non_self_colliding=False)



print(repr(qr), repr(pr))