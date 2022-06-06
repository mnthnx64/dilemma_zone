import colorsys
import cv2
from map_model import MapModel
import numpy as np

def draw_vehicle_obb(map_image, heading, pos, dim, color, thickness=0):
    '''
    Draw the oriented bounding box (obb) on the map image.

    NOTE: "heading", "pos", "dim" should be NumPy arrays

    '''

    if thickness == 0:
        thickness = sum(map_image.shape) // 800

    norm_heading = np.linalg.norm(heading)
    assert norm_heading > 1e-3

    heading = heading / norm_heading
    left = np.array((-heading[1], heading[0]))

    a = pos + heading * dim[0] / 2 + left * dim[1] / 2
    b = pos + heading * dim[0] / 2 - left * dim[1] / 2
    c = pos - heading * dim[0] / 2 + left * dim[1] / 2
    d = pos - heading * dim[0] / 2 - left * dim[1] / 2

    xa, ya = int(a[0]), int(a[1])
    xb, yb = int(b[0]), int(b[1])
    xc, yc = int(c[0]), int(c[1])
    xd, yd = int(d[0]), int(d[1])

    cv2.line(map_image, (xa, ya), (xb, yb), color, thickness)
    cv2.line(map_image, (xb, yb), (xd, yd), color, thickness)
    cv2.line(map_image, (xc, yc), (xd, yd), color, thickness)
    cv2.line(map_image, (xc, yc), (xa, ya), color, thickness)

    pass



def check_export_simple_local_map(data_folder, map_folder, map_name, track):
    '''
    Check the exported vehicle states in local map for plotting.
    '''

    map_vis_scale = 0.7

    wait = -1

    mm = MapModel.construct_from_file(map_folder + '/' + map_name)
    map_image = mm.local_map_image

    idx_fn = map_name + '_' + track + '_simple_lmap_index.csv'
    idx_fn = data_folder + '/' + idx_fn

    data_fn = map_name + '_' + track + '_simple_lmap_data.csv'
    data_fn = data_folder + '/' + data_fn

    idx_array = np.loadtxt(idx_fn, delimiter=',', dtype=int)
    data_array = np.loadtxt(data_fn, delimiter=',')

    nr_frames = len(idx_array)

    for frame_id in range(nr_frames - 1):

        map_vis = map_image.copy()

        row_idx = idx_array[frame_id]
        row_idx_end = idx_array[frame_id + 1]

        subarray = data_array[row_idx:row_idx_end]

        nr_vehicles =  len(subarray)
        for i in range(nr_vehicles):

            vid_i = int(subarray[i, 1])
            pos_i = subarray[i, 2:4]
            heading_i = subarray[i, 4:6]
            dim_i = subarray[i, 6:8]

            # NOTE: Using 19 colors.
            hue = (4 * vid_i) % 19 / 19
            r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
            color_i = r * 255, g * 255, b * 255

            draw_vehicle_obb(map_vis, heading_i, pos_i, dim_i, color_i)


        mh = int(map_vis.shape[0] * map_vis_scale)
        mw = int(map_vis.shape[1] * map_vis_scale)
        mvis_size = (mw, mh)

        map_vis = cv2.resize(map_vis, mvis_size)

        cv2.imshow('map_vis', map_vis)

        c = cv2.waitKey(wait)
        if c & 0xFF == ord('q'):
            break

        



    pass


if __name__ == '__main__':

    data_folder = 'data/metadata/original'
    out_folder = 'data/metadata/simple'
    map_folder = 'data/metadata/2D_maps'

    map_name = 'D0'
    track = '0'

    #export_simple_local_map(data_folder, out_folder,
    #    map_folder, map_name, track)

    check_export_simple_local_map(out_folder,
        map_folder, map_name, track)


    pass