import enum
import numpy as np
import unittest
from klampt_robot import PR2, BaxterArm

class TestPR2(unittest.TestCase):
    def setUp(self):
        self.robot = PR2()
        
    def test_get_active_joint_names(self):
        self.assertEqual(self.robot.get_active_joint_names(), ['l_shoulder_pan_link', 'l_shoulder_lift_link', 'l_upper_arm_roll_link', 'l_elbow_flex_link', 'l_forearm_roll_link', 'l_wrist_flex_link', 'l_wrist_roll_link', 'l_gripper_palm_link'])
    
    def test__Q_to_Qvec(self):
        Q = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        
        ans = np.zeros((1, self.robot._all_joint_limits.shape[0]))

        for i, j in enumerate(self.robot._active_joint_idx):
            ans[0, j] = Q[0, i]
        
        np.testing.assert_array_equal(self.robot._Q_to_Qvec(Q), ans)
        
class TestBaxterArm(unittest.TestCase):
    def setUp(self):
        self.robot = BaxterArm()
        
    def test_forward_kinematics_klampt(self):
         
        Q = np.array([[-1.0872739 ,  0.6491925 , -0.99184623,  2.27386988,  1.53084618, 0.92459961,  2.30449416],[ 0.22884606, -1.97841536, -0.78417624,  0.71994797,  2.04743377,  0.72530155, -2.81696174],[-0.18547875, -1.20406619, -2.44048681,  0.3055426 , -1.98087421, -0.7894862 , -1.03212155],[ 0.50009236, -0.32488407,  2.821617  ,  1.58782899, -0.40976284,  0.17618989, -0.09140851],[-0.52576903, -1.79216131,  1.00635752,  1.6135499 , -2.66022937,  0.67693808, -1.75302975]])  
        Pans = np.array([[-0.16822442,  0.04313561,  0.05650815,  0.21466062,  0.59048799, -0.62277572, -0.46625654],[ 0.24242598,  0.24023615,  1.30156787,  0.96960093, -0.16199757, -0.06125573, -0.17285415],[ 0.44014173,  0.16105797,  1.24671986,  0.6796619 ,  0.47265209,  0.11033818,  0.54998654],[-0.11076718,  0.47755718,  1.09363983,  0.34689041,  0.3038278 ,  0.06995597, -0.88456875],[ 0.27640937,  0.82460814,  0.95220267,  0.15114884,  0.1529556 ,  0.38501375,  0.89750935]])
        
        P = self.robot.forward_kinematics_klampt(Q)
        
        np.testing.assert_allclose(P, Pans, atol=1e-5)
        
if __name__ == '__main__':
    unittest.main()
    