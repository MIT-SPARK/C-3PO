import sys, os
import cv2
import numpy as np
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..','third_party','apolloscape','dataset-api'))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..','third_party','apolloscape','dataset-api','utils'))

import apollo_utils as uts


def get_stereo_rectify_params():
    """Get rectification parameters

    :return:
    """
    # fx, fy, cx, cy
    _data_config = {}
    _data_config['image_size'] = [2710, 3384]
    _data_config['intrinsic'] = {
        'Camera_5': np.array(
            [2304.54786556982, 2305.875668062,
             1686.23787612802, 1354.98486439791]),
        'Camera_6': np.array(
            [2300.39065314361, 2301.31478860597,
             1713.21615190657, 1342.91100799715])}

    camera_names = _data_config['intrinsic'].keys()
    camera5_mat = uts.intrinsic_vec_to_mat(
        _data_config['intrinsic']['Camera_5'])
    camera6_mat = uts.intrinsic_vec_to_mat(
        _data_config['intrinsic']['Camera_6'])

    distCoeff = np.zeros(4)
    image_size = (_data_config['image_size'][1],
                  _data_config['image_size'][0])

    # relative pose of camera 6 wrt camera 5
    _data_config['extrinsic'] = {
        'R': np.array([
            [9.96978057e-01, 3.91718762e-02, -6.70849865e-02],
            [-3.93257593e-02, 9.99225970e-01, -9.74686202e-04],
            [6.69948100e-02, 3.60985263e-03, 9.97746748e-01]]),
        #'T': np.array([-0.6213358, 0.02198739, -0.01986043])
        #'R': np.array([[0.996797, 0.0384542, -0.0701246],
        #               [-0.038475, 0.999259, 0.00105552],
        #               [0.0701132, 0.0016459, 0.997538]]),
        'T': np.array([-0.636491, -0.0158574, -0.0599776])
        # 'T': np.array([-0.706491, -0.0158574, -0.0599776])
    }

    # compare the two image
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=camera5_mat,
        distCoeffs1=distCoeff,
        cameraMatrix2=camera6_mat,
        distCoeffs2=distCoeff,
        imageSize=image_size,
        R=_data_config['extrinsic']['R'],
        T=_data_config['extrinsic']['T'],
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=1)

    Q = np.array([[1, 0, 0, -1489.8536],
                  [0, 1, 0, -479.1750],
                  [0, 0, 0, 2301.3147],
                  [0, 0, 1/np.linalg.norm(_data_config['extrinsic']['T']), 0]])

    #Q = np.array([[1, 0, 0, -1489.8536],
    #              [0, 1, 0, -479.1750],
    #              [0, 0, 0, 2301.3147],
    #              [0, 0, 1/0.68464003, 0]])
    #Q = np.array([[1, 0, 0, -1489.8536],
    #              [0, 1, 0, -479.1750],
    #              [0, 0, 0, 2330.96],
    #              [0, 0, 1/0.68464003, 0]])

    # Load H matrix
    # croppedC_car_rays = H @ unrectC_car_rays
    H_homo = np.array([[ 0.99921196,  0.06244959, -0.0361615 ],
                       [-0.05788579,  0.99849768, -0.22393795],
                       [-0.02417175, -0.00192319, 1.        ]])

    H_image = np.array([[ 9.64832598e-01,  6.08453259e-02, -3.28331026e+02],
                         [-6.16411651e-02,  9.78239378e-01, -1.25720075e+03],
                         [-1.01075359e-05, -8.23664591e-07,  1.00000000e+00]])
    
    # Load rotation matrix estimated by GNC
    # croppedC_car_rays = R_gnc @ unrectC_car_rays
    R_gnc = np.array([[ 0.99812342,  0.05845328, -0.01824439],
                      [-0.06106413,  0.972343  , -0.22543349],
                      [ 0.00456248,  0.22612453,  0.97408772]])

    stereo_params = {'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2, 'Q': Q,
                     'H_homo': H_homo, 'H_image': H_image, 'H_gnc': R_gnc}
    return stereo_params
