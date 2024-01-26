import copy
import os

import cv2
import gsdevice
import numpy as np
import torch
from gs_utils import RGB2NormNet, demark, poisson_dct_neumaan
from sklearn.neighbors import NearestNeighbors


def get_duplicate_id(arr):
    """
    return: list of tuples like (1,[0,1,2]), meaning that index 0,1,2 all have value of 1
    """
    uniq = np.unique(arr).tolist()
    ret = []

    def check_id(target, a):
        b = []
        for index, nums in enumerate(a):
            if nums == target:
                b.append(index)
        return (b)

    for i in range(len(uniq)):
        ret.append([])
    for index, nums in enumerate(arr):
        id = uniq.index(nums)
        ret[id].append(index)

    ans = [(uniq[i], ret[i]) for i in range(len(uniq))]
    return ans


def get_mapping(prev_markers, markers, max_distance: float = 2.5 / 0.04 / 2):
    """获取两帧的关键点之间的映射及丢失情况"""
    prev_markers = copy.deepcopy(prev_markers)
    markers = copy.deepcopy(markers)

    prev_markers_array = np.array(prev_markers)
    markers = np.array(markers)
    if prev_markers_array.ndim <= 1:
        return np.zeros((markers.shape[0],)).astype(int), np.ones((markers.shape[0],)).astype(bool),
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(prev_markers_array)
    distances, indices = nbrs.kneighbors(markers)
    lost = distances > max_distance
    distances = distances.flatten()
    mapping = indices.flatten()
    lost = lost.flatten()

    dup = get_duplicate_id(mapping)
    for (value, index_list) in dup:
        min_id = np.argmin(distances[index_list])
        for duplicated in index_list:
            if duplicated != index_list[min_id]:
                lost[duplicated] = True

    return mapping, lost


class GelSightMini:
    def __init__(self,
                 serial_num: str,
                 marker_threshold=0.4, marker_min_dist=12, marker_min_area=8,
                 blur_size_up=13, blur_size_down=7, blur_size_diff=5,
                 tracking_max_distance=2.5 / 0.04 / 2,
                 contact_depth_threshold=1.0, contact_percentage_threshold=0.08,
                 non_contact_frame_max=10,
                 contact_movement_threshold=15,
                 auto_init_marker_tracker=True,
                 ):
        self.serial_num = serial_num
        self.gs = gsdevice.Camera(serial_num)
        self.gs.connect()  # contact with camera

        self.marker_threshold = marker_threshold
        self.marker_min_dist = marker_min_dist
        self.marker_min_area = marker_min_area
        self.blur_size_up = blur_size_up
        self.blur_size_down = blur_size_down
        self.blur_size_diff = blur_size_diff

        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = 1  # binarization threshold down
        params.maxThreshold = 12  # binarization threshold up
        params.minDistBetweenBlobs = self.marker_min_dist
        params.filterByArea = True 
        params.minArea = self.marker_min_area
        params.maxArea = 300
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.minInertiaRatio = 0.5

        params.thresholdStep = 0.5
        params.minCircularity = 0.1
        params.maxThreshold = 20
        params.filterByCircularity = False
        params.filterByInertia = False
        params.minInertiaRatio = 0.8
        params.minArea = 25
        params.maxArea = 400

        # size of gaussian filter kernal core
        self.blur_size_diff = 15
        self.blur_size_up = 15
        self.blur_size_down = 9

        # gray 2 binary threshold
        self.marker_threshold = 76 / 255

        self.marker_detector = cv2.SimpleBlobDetector_create(params)

        self.tracking_max_distance = tracking_max_distance

        # initialize marker tracker
        f0 = self.gs.get_raw_image()
        # exit()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._init_marker_tracker(f0)

        # depth
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        net_filename = "nnmini.pt"
        nnet_file = os.path.join(curr_dir, net_filename)
        checkpoint = torch.load(nnet_file, map_location=lambda storage, loc: storage)
        self.depth_net = RGB2NormNet().float()
        self.depth_net.load_state_dict(checkpoint['state_dict'])
        self.depth_net = self.depth_net.cuda()
        self.depth_net.eval()

        self.depth_zero = self.compute_depth_map(f0)

        # contact
        self.contact_depth_threshold = contact_depth_threshold
        self.contact_percentage_threshold = contact_percentage_threshold
        self.non_contact_frame_max = non_contact_frame_max
        self.non_contact_frame = 0

        self.contact_movement_threshold = contact_movement_threshold
        self.contact_movement = False

        self.auto_init_marker_tracker = auto_init_marker_tracker

    def _init_marker_tracker(self, frame_image):
        frame_image = self.crop(frame_image)
        marker_mask = self._mask_marker(frame_image)
        self.curr_mask_marker = marker_mask.copy()
        self.init_markers = self._get_markers(marker_mask)
        self.total_mapping = np.arange(len(self.init_markers))
        self.total_lost = np.zeros((len(self.init_markers),)).astype(bool)
        self.last_markers = self.init_markers

    def _mask_marker(self, image):
        """Mask the markers in the image"""
        row, col = image.shape[:2]
        image = image.astype(np.float32)
        blur = cv2.GaussianBlur(image, (self.blur_size_up, self.blur_size_up), 0)
        blur2 = cv2.GaussianBlur(image, (self.blur_size_down, self.blur_size_down), 0)
        diff = blur - blur2

        diff *= 16.0
        diff = np.clip(diff, 0., 255.0)

        diff = cv2.GaussianBlur(diff, (self.blur_size_diff, self.blur_size_diff), 0)

        # cv2.imshow('diff',diff)
        # cv2.waitKey(0)

        marker_mask = diff.sum(-1) > 255 * self.marker_threshold
        marker_mask = marker_mask.astype(np.uint8) * 255
        marker_mask = cv2.dilate(marker_mask, self.kernel, iterations=4)
        marker_mask = cv2.erode(marker_mask, self.kernel, iterations=4)
        # print(marker_mask)
        return marker_mask  # 255 for markers

    def crop(self, image):
        return image



    def _get_markers(self, marker_mask_image):
        "连通域or色块检测"
        marker_points = self.marker_detector.detect(255 - marker_mask_image)
        
        marker_points = [(a.pt[0], a.pt[1]) for a in marker_points]

        # import matplotlib.pyplot as plt

        # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(marker_mask_image, connectivity=4)
        # marker_points = centroids
        # # the first point refer to background
        # marker_points = np.delete(marker_points,0,0)
        return marker_points

    def track_marker(self, frame_image):
        curr_mask_marker = self._mask_marker(frame_image)
        self.curr_mask_marker = curr_mask_marker.copy()

        curr_markers = self._get_markers(curr_mask_marker)
        curr_mapping, curr_lost = get_mapping(self.last_markers, curr_markers, self.tracking_max_distance)

        self.total_mapping = self.total_mapping[curr_mapping]
        self.total_lost = self.total_lost[curr_mapping] | curr_lost
        self.last_markers = curr_markers

        return curr_markers, self.total_mapping, self.total_lost

    def compute_depth_map(self, frame_image):
        ''' Set the contact mask to everything but the markers '''

        # m.init(mc)
        cm1 = (self.curr_mask_marker < 127)

        ''' Get depth image with NN '''
        nx = np.zeros(frame_image.shape[:2])
        ny = np.zeros(frame_image.shape[:2])
        dm = np.zeros(frame_image.shape[:2])

        ''' ENTIRE CONTACT MASK THRU NN '''
        rgb = frame_image[np.where(cm1)] / 255.
        pxpos = np.vstack(np.where(cm1)).T
        pxpos[:, 0], pxpos[:, 1] = pxpos[:, 0] / frame_image.shape[0], pxpos[:, 1] / frame_image.shape[1]
        features = np.column_stack((rgb, pxpos))
        features = torch.from_numpy(features).float().cuda()

        with torch.no_grad():
            out = self.depth_net(features)

        nx[np.where(cm1)] = out[:, 0].cpu().detach().numpy()
        ny[np.where(cm1)] = out[:, 1].cpu().detach().numpy()

        '''OPTION#2 calculate gx, gy from nx, ny. '''
        nz = np.sqrt(np.clip(1 - nx ** 2 - ny ** 2, 1e-6, 1))
        if np.isnan(nz).any():
            print('nan found')

        nz = np.nan_to_num(nz)
        gx = -np.divide(nx, nz)
        gy = -np.divide(ny, nz)

        gx_interp, gy_interp = demark(gx, gy, (self.curr_mask_marker > 127))

        dm1 = poisson_dct_neumaan(gx_interp, gy_interp)
        dm1 = np.array(dm1)
        return dm1

    def inference(self, is_boost: bool = False):
        f = self.gs.get_image()
        f = self.crop(f)

        if self.non_contact_frame >= self.non_contact_frame_max and self.auto_init_marker_tracker:
            self._init_marker_tracker(f)
            self.non_contact_frame = 0

        if is_boost:
            self._init_marker_tracker(f)

        infer_dict = {'current_frame': f.copy()}
        curr_marker, _, _ = self.track_marker(f)
        infer_dict['non_contact_frame'] = self.non_contact_frame
        infer_dict['marker_image'] = self.curr_mask_marker.copy()

        init_markers = self.init_markers
        lost = self.total_lost
        mapping = self.total_mapping
        current_valid_marker = []
        init_valid_marker = []
        for i in range(len(curr_marker)):
            if not lost[i]:
                init_valid_marker.append([init_markers[mapping[i]][0], init_markers[mapping[i]][1]])
                current_valid_marker.append([curr_marker[i][0], curr_marker[i][1]])

        infer_dict['curr_marker'] = np.array(current_valid_marker)
        infer_dict['init_marker'] = np.array(init_valid_marker)

        depth = self.compute_depth_map(f)
        depth -= self.depth_zero
        infer_dict['depth'] = depth

        contact_mask = (depth > self.contact_depth_threshold)
        if contact_mask.mean() < self.contact_percentage_threshold:
            self.non_contact_frame += 1
            self.contact = False
            self.contact_movement = False
        else:
            self.non_contact_frame = 0
            self.contact = True
        infer_dict['contact_depth'] = self.contact

        init_pts = []
        curr_pts = []
        for i in range(len(curr_marker)):
            marker = curr_marker[i]
            if contact_mask[int(marker[1]), int(marker[0])] and not self.total_lost[i]:
                init_pts.append(np.array([self.init_markers[self.total_mapping[i]][0],
                                          self.init_markers[self.total_mapping[i]][1]]))
                curr_pts.append(np.array([marker[0], marker[1]]))

        num_contact_markers = len(curr_pts)
        if num_contact_markers < 2:
            self.contact_movement = False
        else:
            init_pts = np.stack(init_pts)
            curr_pts = np.stack(curr_pts)

            contact_mean_squared_movement = ((curr_pts - init_pts) ** 2).sum(-1).sum() / num_contact_markers
            if contact_mean_squared_movement >= self.contact_movement_threshold ** 2:
                self.contact_movement = True
            else:
                self.contact_movement = False

        infer_dict["contact_movement"] = self.contact_movement

        return infer_dict

    def close(self):
        self.gs.stop_video()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    gelsight = GelSightMini("28F0-GX7F")

    while True:
        infer_dict = gelsight.inference()
        marker_image = infer_dict["marker_image"]
        marker_image = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2BGR)

        curr_marker = infer_dict["curr_marker"]
        print("num of marker: ", len(curr_marker))
        curr_marker = np.round(curr_marker).astype(int)
        
        for m in curr_marker:
            cv2.circle(marker_image, (m[0],m[1]), 10, (0,0,255))

        cv2.imwrite("marker_image.png", marker_image)
        cv2.imshow("img",marker_image)
        cv2.waitKey(1)
    
    cv2.destroyAllWindows()

    depth = infer_dict["depth"]
    # plt.imshow(depth)
    # plt.colorbar()
    # plt.show()
