'''
Created on May 9, 2020

@author: duolu
'''

import numpy as np
import cv2

import pywgs84

class MapModel(object):
    '''
    The map model describes a local map and coordinate transformations.

    The area covered by the map is slightly larger than the area covered
    by a drone camera flying at the height of 100 meters with roughly 90 degree
    field of view.

    The size of the map image is typically 2880 * 2560 pixels. We also
    use map sizes such as 5120 * 1440, 4320 * 3840 or 7680 * 2160. The image is
    typically obtained by taking a snapshot of satellite map images from online
    map services such as Google Maps or Bing Maps. It is assumed that the map
    image always follows the convention of up-for-north, down-for-south,
    left-for-west, right-for-east. The map image is an ndarray of three color
    channels in memory.

    There are four different reference frames defined.

        (1) The local XYZ frame.
        (2) The local map frame, "lmap" for short, where the origin is the top
            left corner of the local map image and the unit is one map pixel.
        (3) The local ESD frame. It is on a tangent plane of the WGS84 geodetic
            ellipsoid relative to a reference point (which is also the origin of
            the plane). This reference point has known latitude-longitude-height
            values and typically it is a point selected on the local map. The
            unit is one metric meter. Typically this tangent plane is assumed to
            be parallel to the local map 2D plane.
        (4) The WGS84 LLH frame, i.e., (latitude, longitude, height).

    The XYZ frame is usually defined arbitrarily but its z-axis always
    points to the up, i.e., the XOY plane is a flat surface perpendicular to
    the gravity. Typically, we only care transformations between the coordinates
    on the XOY plane and the local map frame of the map image pixels.

    We can convert coordinates between any two frames. Converting coordinates
    from LLH frame or to LLH frame is done using the ESD frame as a bridge.

    Attributes:

        local_map_image(ndarray): The map image.

        R_xyz2lmap(3-by-3 ndarray): The rotation matrix from XYZ to lmap.
        t_xyz2lmap(3-by-1 ndarray): The translation vector from XYZ to lmap.

        R_lmap2xyz(3-by-3 ndarray): The rotation matrix from lamp to XYZ.
        t_lmap2xyz(3-by-1 ndarray): The translation vector from lmap to XYZ.

        R_xyz2esd(3-by-3 ndarray): The rotation matrix from XYZ to ESD.
        t_xyz2esd(3-by-1 ndarray): The translation vector from XYZ to ESD.

        R_esd2xyz(3-by-3 ndarray): The rotation matrix from ESD to XYZ.
        t_esd2xyz(3-by-1 ndarray): The translation vector from ESD to XYZ.

        R_esd2lmap(3-by-3 ndarray): The rotation matrix from ESD to lmap.
        t_esd2lmap(3-by-1 ndarray): The translation vector from ESD to lmap.

        R_lmap2esd(3-by-3 ndarray): The rotation matrix from lamp to ESD.
        t_lmap2esd(3-by-1 ndarray): The translation vector from lmap to ESD.

        ref_point_llh(ndarray): The LLH coordinates of the origin of XYZ frame.

    '''

    def __init__(self):
        '''
        Initialize the map model.

        **NOTE**: The transformation is set to identity by default. Typically,
        after the map is initialized, we use "load_map_para()" to set up the
        transformations and we use "load_map_image()" to set up the map image.

        '''

        self.local_map_image = None

        self.R_xyz2lmap = np.identity(3)
        self.t_xyz2lmap = np.zeros(3)

        self.R_lmap2xyz = np.identity(3)
        self.t_lmap2xyz = np.zeros(3)

        self.R_xyz2esd = np.identity(3)
        self.t_xyz2esd = np.zeros(3)

        self.R_esd2xyz = np.identity(3)
        self.t_esd2xyz = np.zeros(3)

        self.R_esd2lmap = np.identity(3)
        self.t_esd2lmap = np.zeros(3)

        self.R_lmap2esd = np.identity(3)
        self.t_lmap2esd = np.zeros(3)

        self.ref_point_llh = np.zeros(3)

    def get_local_map_size(self):
        '''
        Get the size of the local map, i.e., (width, height) in pixels.
        '''

        shape = self.local_map_image.shape

        return shape[1], shape[0]


    def calibrate_local_map_to_xyz(self, pp_xyz):
        '''
        Calculate transformation between the local xyz frame and the
        map local frame.

        NOTE: "pp_xyz" is a 2-by-6 matrix. Both rows are point
        correspondences of (xm, ym, zm, x, y, z), where (xm, ym, zm)
        is a point on the map and (x, y, z) is the a point in the 3D space,
        i.e., local XYZ reference frame. Here the first row is the origin
        of the XYZ reference frame, and the second row is "p1", a point
        on the x-axis of the XYZ reference frame.

        '''

        # Convert to two column vectors
        pp_xyz = pp_xyz.T

        # CAUTION: A point is a 3-by-1 matrix.
        origin_xyz_in_lmap = pp_xyz[0:3, 0].reshape((3, 1))
        px_xyz_in_lmap = pp_xyz[0:3, 1].reshape((3, 1))

        # Obtain the point on the x-axis of the XYZ reference frame.
        # Note that the x coordinate of this point may be positive or
        # negative, but we always want a vector from the origin and
        # pointing along the x-axis.
        point_x = pp_xyz[3, 1]
        if point_x > 0:
            vector_x_in_lmap = px_xyz_in_lmap - origin_xyz_in_lmap
        else:
            vector_x_in_lmap = origin_xyz_in_lmap - px_xyz_in_lmap

        length_on_lmap = np.linalg.norm(vector_x_in_lmap)
        length_in_xyz = np.linalg.norm(pp_xyz[3:6, 1])

        # print(length_on_lmap, length_in_xyz)

        # Here "n_x", "n_y", and "n_z" are the three normalized vectors
        # of the XYZ axes in local map pixel coordinates.
        # CAUTION: In the XYZ frame, "z" means up,
        # while in the local map frame, "d" means down
        n_x = vector_x_in_lmap / np.linalg.norm(vector_x_in_lmap)
        n_y = np.asarray((n_x[1], -n_x[0], 0)).reshape((3, 1))
        n_z = np.asarray((0, 0, -1)).reshape((3, 1))

        # scale conversion

        scale_xyz2lmap = length_on_lmap / length_in_xyz
        scale_lmap2xyz = length_in_xyz / length_on_lmap

        # from xyz to lmap
        # Here "rm" means "rotation matrix".

        r = np.zeros((3, 3))
        r[:, 0:1] = n_x
        r[:, 1:2] = n_y
        r[:, 2:3] = n_z

        R_xyz2lmap = r * scale_xyz2lmap
        t_xyz2lmap = origin_xyz_in_lmap

        # from lmap to xyz

        r_inv = r.T

        R_lmap2xyz = r_inv * scale_lmap2xyz
        t_lmap2xyz = -np.matmul(r_inv, t_xyz2lmap) * scale_lmap2xyz

        # from xyz to esd
        # NOTE: There is only a rotation between xyz and esd, since the
        # anchor point of esd frame is the origin of xyz.

        R_xyz2esd = r
        t_xyz2esd = np.zeros(3)

        # from esd to xyz

        R_esd2xyz = r_inv
        t_esd2xyz = np.zeros(3)

        # from esd to lmap
        # Here "srm" means "scaled rotation matrix".

        srm = np.eye(3) * scale_xyz2lmap

        R_esd2lmap = srm
        t_esd2lmap = t_xyz2lmap

        # from lmap to esd

        S_inv = np.eye(3) * scale_lmap2xyz

        R_lmap2esd = S_inv
        t_lmap2esd = t_lmap2xyz


        # keep calibration results

        self.R_xyz2lmap = R_xyz2lmap
        self.t_xyz2lmap = t_xyz2lmap

        self.R_lmap2xyz = R_lmap2xyz
        self.t_lmap2xyz = t_lmap2xyz

        self.R_xyz2esd = R_xyz2esd
        self.t_xyz2esd = t_xyz2esd

        self.R_esd2xyz = R_esd2xyz
        self.t_esd2xyz = t_esd2xyz

        self.R_esd2lmap = R_esd2lmap
        self.t_esd2lmap = t_esd2lmap

        self.R_lmap2esd = R_lmap2esd
        self.t_lmap2esd = t_lmap2esd


    def calibrate_local_map_to_wgs(self, pp_wgs):
        '''
        Calculate transformation between the local ESD frame to WGS84 frame.

        NOTE: "pp_wgs" has only one row (xm, ym, zm, x, y, z), where the point
        (xm, ym, zm) is the origin of the XYZ reference frame.
        '''

        _ref_point_lmap = pp_wgs[0:3]
        ref_point_llh = pp_wgs[3:6]

        # The XYZ reference frame and the ESD reference frame have the
        # same origin and this point is the reference point of the ESD
        # tangent plane in WGS84.

        self.ref_point_llh = ref_point_llh

    def load_map_image(self, folder):
        '''
        Load the map image. It always has the name "map_local.png".
        '''

        fn = folder + '/seg/map_local_seg.png'
        self.local_map_image = cv2.imread(fn)

    def save_one_para_csv(self, folder, para_name):
        '''
        Save a single set of parameters into a file.
        '''

        para = getattr(self, para_name)
        fn = folder + '/' + para_name + '.csv'
        np.savetxt(fn, para, delimiter=',', fmt='%.6f')

    def load_one_para_csv(self, folder, para_name):
        '''
        Load a single set of parameters from a file.
        '''

        fn = folder + '/' + para_name + '.csv'
        para = np.loadtxt(fn, delimiter=',')
        setattr(self, para_name, para)

    def save_map_para(self, folder):
        '''
        Save the map transformation parameters as files.

        **NOTE**: Each rotation matrix or translation vector is stored
        individually as a CSV file.

        '''

        self.save_one_para_csv(folder, 'R_xyz2lmap')
        self.save_one_para_csv(folder, 't_xyz2lmap')
        self.save_one_para_csv(folder, 'R_lmap2xyz')
        self.save_one_para_csv(folder, 't_lmap2xyz')

        self.save_one_para_csv(folder, 'R_xyz2esd')
        self.save_one_para_csv(folder, 't_xyz2esd')
        self.save_one_para_csv(folder, 'R_esd2xyz')
        self.save_one_para_csv(folder, 't_esd2xyz')

        self.save_one_para_csv(folder, 'R_esd2lmap')
        self.save_one_para_csv(folder, 't_esd2lmap')
        self.save_one_para_csv(folder, 'R_lmap2esd')
        self.save_one_para_csv(folder, 't_lmap2esd')

        self.save_one_para_csv(folder, 'ref_point_llh')


    def load_map_para(self, folder):
        '''
        Load the map transformation parameters from files.

        **NOTE**: Each rotation matrix or translation vector is stored
        individually as a CSV file.

        '''

        self.load_one_para_csv(folder, 'R_xyz2lmap')
        self.load_one_para_csv(folder, 't_xyz2lmap')
        self.load_one_para_csv(folder, 'R_lmap2xyz')
        self.load_one_para_csv(folder, 't_lmap2xyz')

        self.load_one_para_csv(folder, 'R_xyz2esd')
        self.load_one_para_csv(folder, 't_xyz2esd')
        self.load_one_para_csv(folder, 'R_esd2xyz')
        self.load_one_para_csv(folder, 't_esd2xyz')

        self.load_one_para_csv(folder, 'R_esd2lmap')
        self.load_one_para_csv(folder, 't_esd2lmap')
        self.load_one_para_csv(folder, 'R_lmap2esd')
        self.load_one_para_csv(folder, 't_lmap2esd')

        self.load_one_para_csv(folder, 'ref_point_llh')

        # NOTE: When the translation vectors are saved in the files, they are
        # saved in just a single line. Here we need to convert them back to
        # column vectors.

        self.t_xyz2lmap = self.t_xyz2lmap.reshape((3, 1))
        self.t_lmap2xyz = self.t_lmap2xyz.reshape((3, 1))

        self.t_xyz2esd = self.t_xyz2esd.reshape((3, 1))
        self.t_esd2xyz = self.t_esd2xyz.reshape((3, 1))

        self.t_esd2lmap = self.t_esd2lmap.reshape((3, 1))
        self.t_lmap2esd = self.t_lmap2esd.reshape((3, 1))


    def transform_point(self, p, r, t):
        '''
        Transform of a 3D point "p" (Eclidean transformation).

        Here "r" means the rotation matrix,
        and "t" means the translation vector.
        '''

        p = p.reshape((3, 1))

        q = np.matmul(r, p) + t

        q = q.reshape(3)

        return q

    def transform_points(self, ps, r, t):
        '''
        Transform of multiple 3D points "ps" (Eclidean transformation).

        NOTE: "ps" must be a 3-by-n matrix for n 3D points.

        Here "r" means the rotation matrix,
        and "t" means the translation vector.
        '''

        qs = np.matmul(r, ps) + t

        return qs

    def transform_point_lmap_to_xyz(self, p_lmap):
        '''
        Transform a point from the local map frame to the XYZ frame.
        '''

        return self.transform_point(p_lmap, self.R_lmap2xyz, self.t_lmap2xyz)

    def transform_points_lmap_to_xyz(self, ps_lmap):
        '''
        Transform multiple points from the local map frame to the XYZ frame.
        '''

        return self.transform_points(ps_lmap, self.R_lmap2xyz, self.t_lmap2xyz)

    def transform_point_xyz_to_lmap(self, p_xyz):
        '''
        Transform a point from the XYZ frame to the local map frame.
        '''

        return self.transform_point(p_xyz, self.R_xyz2lmap, self.t_xyz2lmap)

    def transform_points_xyz_to_lmap(self, ps_xyz):
        '''
        Transform multiple points from the XYZ frame to the local map frame.
        '''

        return self.transform_points(ps_xyz, self.R_xyz2lmap, self.t_xyz2lmap)

    def transform_point_esd_to_xyz(self, p_esd):
        '''
        Transform a point from the ESD frame to the XYZ frame.
        '''

        return self.transform_point(p_esd, self.R_esd2xyz, self.t_esd2xyz)

    def transform_points_esd_to_xyz(self, ps_esd):
        '''
        Transform multiple points from the ESD frame to the XYZ frame.
        '''

        return self.transform_points(ps_esd, self.R_esd2xyz, self.t_esd2xyz)

    def transform_point_xyz_to_esd(self, p_xyz):
        '''
        Transform a point from the XYZ frame to the ESD frame.
        '''

        return self.transform_point(p_xyz, self.R_xyz2esd, self.t_xyz2esd)

    def transform_points_xyz_to_esd(self, ps_xyz):
        '''
        Transform multiple points from the XYZ frame to the ESD frame.
        '''

        return self.transform_points(ps_xyz, self.R_xyz2esd, self.t_xyz2esd)

    def transform_point_esd_to_lmap(self, p_esd):
        '''
        Transform a point from the ESD frame to the local map frame.
        '''

        return self.transform_point(p_esd, self.R_esd2lmap, self.t_esd2lmap)

    def transform_points_esd_to_lmap(self, ps_esd):
        '''
        Transform multiple points from the ESD frame to the local map frame.
        '''

        return self.transform_points(ps_esd, self.R_esd2lmap, self.t_esd2lmap)

    def transform_point_lmap_to_esd(self, p_lmap):
        '''
        Transform a point from the local map frame to the ESD frame.
        '''

        return self.transform_point(p_lmap, self.R_lmap2esd, self.t_lmap2esd)

    def transform_points_lmap_to_esd(self, ps_lmap):
        '''
        Transform multiple points from the local map frame to the ESD frame.
        '''

        return self.transform_points(ps_lmap, self.R_lmap2esd, self.t_lmap2esd)



    def transform_points_esd_to_llh(self, ps_esd):
        '''
        Transform multiple points from the ESD frame to the WGS84 frame.
        '''

        n = ps_esd.shape[1]

        ps_llh = np.zeros((3, n))

        for i in range(n):

            esd = tuple(ps_esd[:, i])
            llh = pywgs84.esd_to_llh(*esd, self.ref_point_llh)
            ps_llh[:, i] = np.asarray(llh)

        return ps_llh

    def transform_points_llh_to_esd(self, ps_llh):
        '''
        Transform multiple points from the WGS84 frame to the ESD frame.
        '''

        n = ps_llh.shape[1]

        ps_esd = np.zeros((3, n))

        for i in range(n):

            llh = tuple(ps_llh[:, i])
            esd = pywgs84.llh_to_esd(*llh, self.ref_point_llh)
            ps_esd[:, i] = np.asarray(esd).flatten()

        return ps_esd


    @classmethod
    def construct_from_file(cls, folder):
        '''
        Construct a map model object from files storing the parameters.
        '''

        map_model = MapModel()
        map_model.load_map_image(folder)
        map_model.load_map_para(folder)

        return map_model


def demo_map_model():
    '''
    Demo code of the usage of the map model class and methods.
    '''

    map_folder = '../carom_air_data/aerial_map_model_2d'
    map_name = 'D0'

    mm = MapModel.construct_from_file(map_folder + '/' + map_name)

    q_xyz = np.array((1, 2, 3))
    q_lmap = mm.transform_point_xyz_to_lmap(q_xyz)
    print(q_lmap)







if __name__ == '__main__':

    demo_map_model()
