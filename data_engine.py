
from sklearn.utils.extmath import cartesian
import numpy as np
from PIL import Image

import NMS
import os

wandhG = [[100.0, 100.0], [300.0, 300.0], [500.0, 500.0],
          [200.0, 100.0], [370.0, 185.0], [440.0, 220.0],
          [100.0, 200.0], [185.0, 370.0], [220.0, 440.0]]

def getAllFiles(dirName, houzhui):
    results = []

    for file in os.listdir(dirName):
        file_path = os.path.join(dirName, file)
        if os.path.isfile(file_path) and os.path.splitext(file_path)[1] == houzhui:
            results.append([file_path,os.path.splitext(file)[0]])

    return results

class RPN_Test(object):
    def __init__(self):

        self.image_height = 720
        self.image_width = 960

        self.convmap_height = int(np.ceil(self.image_height / 16.))
        self.convmap_width = int(np.ceil(self.image_width / 16.))

        self.anchor_size = 9

        self.bbox_normalize_scale = 5
        self.wandh = wandhG
        self.proposal_prepare()

    def rpn_nms(self, prob, bbox_pred):
        prob = prob[:, 0]
        bbox_pred /= self.bbox_normalize_scale
        anchors = self.proposals.copy()
        anchors[:, 2] -= anchors[:, 0]
        anchors[:, 3] -= anchors[:, 1]
        anchors[:, 0] = bbox_pred[:, 0] * anchors[:, 2] + anchors[:, 0]
        anchors[:, 1] = bbox_pred[:, 1] * anchors[:, 3] + anchors[:, 1]
        anchors[:, 2] = np.exp(bbox_pred[:, 2]) * anchors[:, 2]
        anchors[:, 3] = np.exp(bbox_pred[:, 3]) * anchors[:, 3]
        bbox = np.zeros([anchors.shape[0], 5])
        
        bbox[:, :4] = anchors
        bbox[:, 4] = prob
        bbox = NMS.filter_bbox(bbox)
        bbox = NMS.non_max_suppression_fast(bbox, 0.7)

        keep_prob = np.sort(bbox[:, 4])[max(-50, -1 * bbox.shape[0])]
        
        index = np.where(bbox[:, 4] >= keep_prob)[0]
        bbox = bbox[index]
        return bbox

    def proposal_prepare(self):
        
        anchors = self.generate_anchors()
        proposals = np.zeros([self.anchor_size * self.convmap_width * self.convmap_height, 4])

        for i in range(self.convmap_height):
            h = i * 16 + 8
            for j in range(self.convmap_width):
                w = j * 16 + 8
                for k in range(self.anchor_size):
                    index = i * self.convmap_width * self.anchor_size + j * self.anchor_size + k
                    anchor = anchors[k, :]
                    proposals[index, :] = anchor + np.array([w, h, w, h])

        
        self.proposals = proposals

    def generate_anchors(self):
        anchors = np.zeros([self.anchor_size, 4])

        for i in range(self.anchor_size):
            anchor_width = self.wandh[i][0]
            anchor_height = self.wandh[i][1]
            anchors[i, :] = np.array(
                [-0.5 * anchor_width, -0.5 * anchor_height, 0.5 * anchor_width, 0.5 * anchor_height])
        return anchors

class My_Caltech_Test(object):
    def __init__(self ,original):
        self.original = original;
        self.image_height = 720
        self.image_width = 960

        self.convmap_height = int(np.ceil(self.image_height / 16.))
        self.convmap_width = int(np.ceil(self.image_width / 16.))

        self.anchor_size = 9
        self.img_resize = 1.5
        self.bbox_normalize_scale = 5
        self.wandh = wandhG

        self.aspect_ratio = 0.41
        self.image_resize_factor = 1.5
        self.anchor_min_height = 40 * self.image_resize_factor
        self.anchor_factor = 1.3

        self.proposal_prepare()

    def rpn_nms(self, prob, bbox_pred):
        prob = prob[:, 0]
        bbox_pred /= self.bbox_normalize_scale
        
        anchors = self.proposals.copy()
        anchors[:, 2] -= anchors[:, 0]
        anchors[:, 3] -= anchors[:, 1]
        anchors[:, 0] = bbox_pred[:, 0] * anchors[:, 2] + anchors[:, 0]
        anchors[:, 1] = bbox_pred[:, 1] * anchors[:, 3] + anchors[:, 1]
        anchors[:, 2] = np.exp(bbox_pred[:, 2]) * anchors[:, 2]
        anchors[:, 3] = np.exp(bbox_pred[:, 3]) * anchors[:, 3]
        bbox = np.zeros([anchors.shape[0], 5])
        
        bbox[:, :4] = anchors
        bbox[:, 4] = prob
        bbox = NMS.filter_bbox(bbox)
        bbox = NMS.non_max_suppression_fast(bbox, 0.7)

        keep_prob = np.sort(bbox[:, 4])[max(-50, -1 * bbox.shape[0])]
        
        index = np.where(bbox[:, 4] >= keep_prob)[0]
        bbox = bbox[index]

        bbox[:, :4] = bbox[:, :4] / self.img_resize

        return bbox
    
    def proposal_prepare(self):
       
        anchors = self.generate_anchors()
        proposals = np.zeros([self.anchor_size * self.convmap_width * self.convmap_height, 4])

        for i in range(self.convmap_height):
            h = i * 16 + 8
            for j in range(self.convmap_width):
                w = j * 16 + 8
                for k in range(self.anchor_size):
                    index = i * self.convmap_width * self.anchor_size + j * self.anchor_size + k
                    anchor = anchors[k, :]
                    proposals[index, :] = anchor + np.array([w, h, w, h])

        
        self.proposals = proposals

    def generate_anchors(self):
        if self.original:
            anchors = np.zeros([self.anchor_size, 4])
            anchor_height = self.anchor_min_height
            for i in range(self.anchor_size):
                anchor_width = anchor_height * self.aspect_ratio
                anchors[i, :] = np.array(
                    [-0.5 * anchor_width, -0.5 * anchor_height, 0.5 * anchor_width, 0.5 * anchor_height])
                anchor_height *= self.anchor_factor
            return anchors
        else:
            anchors = np.zeros([self.anchor_size, 4])

            for i in range(self.anchor_size):
                anchor_width = self.wandh[i][0]
                anchor_height = self.wandh[i][1]
                anchors[i, :] = np.array(
                    [-0.5 * anchor_width, -0.5 * anchor_height, 0.5 * anchor_width, 0.5 * anchor_height])
            return anchors

    def open(self,imgPath):
        im = Image.open(imgPath)
        return im.resize( ( int(im.width*self.img_resize), int(im.height*self.img_resize) ), Image.ANTIALIAS)

class CNNData(object):
    def __init__(self, batch_size=128, imageLoadDir='' , anoLoadDir='',original = False):
        self.batch_size = batch_size
        if anoLoadDir == '':
            self.useList = True
        else:
            self.useList = False

        if self.useList:
            self.listName = imageLoadDir
        else:
            self.imageLoadDir = imageLoadDir
            self.anoLoadDir = anoLoadDir


        self.aspect_ratio = 0.41
        self.image_resize_factor = 1.5
        self.image_height = 720
        self.image_width = 960

        self.convmap_height = int(np.ceil(self.image_height / 16.))
        self.convmap_width = int(np.ceil(self.image_width / 16.))

        self.anchor_min_height = 40 * self.image_resize_factor
        self.anchor_factor = 1.3
        self.anchor_size = 9

        self.fg_thresh = 0.5
        self.bg_thresh = 0.2

        self.bbox_normalize_scale = 5
        self.wandh = wandhG

        self.original = original

        self.load_data()

    def load_data(self):
        
        print ('Load Training Data')

        self.imdb_train = self.load_image()
        self.imdb_train = self.proposal_prepare(self.imdb_train)
        print ('Done')

        self.inds = self.generate_minibatch()
        print ('Total Batches:', self.inds.shape[0])

        self.idx = 0

    def load_test_data(self, testDataPath):
        
        print ('Load Testing Data')

        print ('Done')

    def prepare_data(self):
        if self.idx == self.inds.shape[0]:
            self.inds = self.generate_minibatch()
            self.idx = 0

        ind = self.inds[self.idx]
        im_train = self.imdb_train[ind]
        self.idx += 1

        im = Image.open(im_train['name'])
        pix = np.array(im.getdata()).reshape(1, self.image_height, self.image_width, 3).astype(np.float32)

        roi_anchor = im_train['roi_anchor']

        anchors_size = roi_anchor.shape[0]

        labels = np.hstack([np.zeros([anchors_size, 1]), np.ones([anchors_size, 1])])
        fg_idx = np.where(roi_anchor[:, 0] == 1)[0]
        
        bg_idx = np.where(roi_anchor[:, 0] == -1)[0]
        labels[fg_idx, 0] = 1
        labels[fg_idx, 1] = 0
        bbox_targets = roi_anchor[:, 1:5] * self.bbox_normalize_scale

        
        fg_num = min(fg_idx.shape[0], self.batch_size / 6)
        np.random.shuffle(fg_idx)
        fg_idx = fg_idx[:fg_num]
        bg_num = min(self.batch_size - fg_num, 5 * fg_num)
        np.random.shuffle(bg_idx)
        bg_idx = bg_idx[:bg_num]

        labels_weight = np.zeros(anchors_size)
        bbox_loss_weight = np.zeros(anchors_size)
        labels_weight[fg_idx] = 1
        labels_weight[bg_idx] = 1
        bbox_loss_weight[fg_idx] = 1

        return pix, labels, labels_weight, bbox_targets, bbox_loss_weight

    def get_testdata_size(self):
        return len(self.imdb_test)

    def prepare_test_data(self, idx):
        assert (idx >= 0)
        assert (idx < len(self.imdb_test))
        im_test = self.imdb_test[idx]
        im = Image.open(self.path + 'test/images_resize/' + im_test['name'] + '.jpg')
        pix = np.array(im.getdata()).reshape(1, self.image_height, self.image_width, 3).astype(np.float32)
        return pix

    def post_process(self, idx, prob, bbox_pred):
        prob = prob[:, 0]
        bbox_pred /= self.bbox_normalize_scale
        keep_prob = np.sort(prob)[-1000]
        index = np.where(prob >= keep_prob)[0]
        anchors = self.proposals.copy()
        anchors[:, 2] -= anchors[:, 0]
        anchors[:, 3] -= anchors[:, 1]
        anchors[:, 0] = bbox_pred[:, 0] * anchors[:, 2] + anchors[:, 0]
        anchors[:, 1] = bbox_pred[:, 1] * anchors[:, 3] + anchors[:, 1]
        anchors[:, 2] = np.exp(bbox_pred[:, 2]) * anchors[:, 2]
        anchors[:, 3] = np.exp(bbox_pred[:, 3]) * anchors[:, 3]
        self.imdb_test[idx]['bbox'] = anchors[index, :]
        self.imdb_test[idx]['prob'] = prob[index]

    def save_test(self, iter, save_dir):
        n = len(self.imdb_test)
        f = open(save_dir + 'RPN_' + str(iter) + '.txt', 'w')
        for i in range(n):
            im_test = self.imdb_test[i]
            bbox_num = im_test['prob'].shape[0]
            for j in range(bbox_num):
                f.write(str(i + 1) + ' ')
                for k in range(4):
                    f.write(str(im_test['bbox'][j, k]) + ' ')
                f.write(str(im_test['prob'][j]) + '\n')
        f.close()

    def getImgAndAnoFromList(self, listName):
        res = []
        f = open(listName, "r")
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            ss = line.split(' ')
            if len(ss) == 2:
                res.append(ss)
        return res

    def load_image(self,flip = 0):
        imdb = []
        if self.useList:
            self.files = self.getImgAndAnoFromList(self.listName)
            for fileNow in self.files:
                roi = self.load_roi(fileNow[1])
                iminfo = {'name': fileNow[0], 'image': None, 'roi': roi}
                imdb.append(iminfo)
                if flip:
                    roi_f = self.flip_roi(roi)
                    iminfo = {'name': fileNow[0] + '_flip', 'image': None, 'roi': roi_f}
                    imdb.append(iminfo)
        else:
            self.files = getAllFiles(self.imageLoadDir, '.jpg')
            for fileNow in self.files:
                roi = self.load_roi(self.anoLoadDir + '/' + fileNow[1] + '.txt')
                iminfo = {'name': fileNow[0], 'image': None, 'roi': roi}
                imdb.append(iminfo)
                if flip:
                    roi_f = self.flip_roi(roi)
                    iminfo = {'name': fileNow[0] + '_flip', 'image': None, 'roi': roi_f}
                    imdb.append(iminfo)

        return imdb

    def load_roi(self, path):
        f = open(path)
        bbs = f.readlines()[1:]
        roi = np.zeros([len(bbs), 5])
        for iter_, bb in zip(range(len(bbs)), bbs):
            bb = bb.replace('\n', '').split(' ')
            bbtype = bb[0]
            bba = np.array([float(bb[i]) for i in range(1, 5)])
            occ = float(bb[5])
            bbv = np.array([float(bb[i]) for i in range(6, 10)])
            ignore = int(bb[10])

            ignore = ignore or (bbtype != 'person')
            ignore = ignore or (bba[3] < 40)

           

            roi[iter_, :4] = bba
            roi[iter_, 4] = ignore
        return roi

    def flip_roi(self, roi):
        roi_f = np.zeros(roi.shape)
        for i in range(roi_f.shape[0]):
            roi_f[i, :] = roi[i, :]
            roi_f[i, 0] = self.image_width - roi[i, 0] - roi[i, 2]
        return roi_f

    def generate_anchors(self):
        if self.original:
            anchors = np.zeros([self.anchor_size, 4])
            anchor_height = self.anchor_min_height
            for i in range(self.anchor_size):
                anchor_width = anchor_height * self.aspect_ratio
                anchors[i, :] = np.array(
                    [-0.5 * anchor_width, -0.5 * anchor_height, 0.5 * anchor_width, 0.5 * anchor_height])
                anchor_height *= self.anchor_factor
            return anchors
        else:
            anchors = np.zeros([self.anchor_size, 4])

            for i in range(self.anchor_size):
                anchor_width = self.wandh[i][0]
                anchor_height = self.wandh[i][1]
                anchors[i, :] = np.array(
                    [-0.5 * anchor_width, -0.5 * anchor_height, 0.5 * anchor_width, 0.5 * anchor_height])
            return anchors

    def proposal_prepare(self, imdb):
      
        anchors = self.generate_anchors()
        proposals = np.zeros([self.anchor_size * self.convmap_width * self.convmap_height, 4])

        for i in range(self.convmap_height):
            h = i * 16 + 8
            for j in range(self.convmap_width):
                w = j * 16 + 8
                for k in range(self.anchor_size):
                    index = i * self.convmap_width * self.anchor_size + j * self.anchor_size + k
                    anchor = anchors[k, :]
                    proposals[index, :] = anchor + np.array([w, h, w, h])

        # ignore cross-boundary anchors
        self.proposals = proposals
        proposals_keep = np.where(
            (proposals[:, 0] > -5) & (proposals[:, 1] > -5) & (proposals[:, 2] < self.image_width + 5) & (
                proposals[:, 3] < self.image_height + 5))[0]
        self.proposals_mask = np.zeros(proposals.shape[0])
        self.proposals_mask[proposals_keep] = 1

        area = (proposals[:, 2] - proposals[:, 0]) * (proposals[:, 3] - proposals[:, 1])
        proposals = np.hstack([proposals, area.reshape([area.shape[0], 1])])
       
        n = len(imdb)
        foreground_anchor_size = np.zeros(n)
        for i in range(n):
            imdb[i]['roi_anchor'], foreground_anchor_size[i] = compute_target(imdb[i]['roi'], proposals, self.fg_thresh,
                                                                              self.bg_thresh)
            imdb[i]['fgsize']= foreground_anchor_size[i]
            if i % 500 == 0:
                print('Compute Target: %d/%d' % (i, n))
        print('Compute Target: %d/%d' % (n, n))
        self.fg_anchors_per_image = foreground_anchor_size

        return imdb

    def generate_minibatch(self):
        keep = np.where(self.fg_anchors_per_image >= 10)[0]
        np.random.shuffle(keep)
        return keep

def compute_target(roi_t, proposals, fg_thresh, bg_thresh):
    roi = roi_t.copy()
    roi[:, 2] += roi[:, 0]
    roi[:, 3] += roi[:, 1]
    proposal_size = proposals.shape[0]
    roi_anchor = np.zeros([proposal_size, 5])

    if roi.shape[0] == 0:
        return roi_anchor, 0

    overlap = compute_overlap(roi, proposals)
    overlap_max = np.max(overlap, axis=1)
    overlap_max_idx = np.argmax(overlap, axis=1)

    for i in range(proposal_size):
        if overlap_max[i] >= fg_thresh:
            if roi[overlap_max_idx[i], 4] == 0:
                roi_anchor[i, 0] = 1
                roi_anchor[i, 1:5] = compute_regression(roi[overlap_max_idx[i], :4], proposals[i, :])
        if overlap_max[i] <= bg_thresh:
            roi_anchor[i, 0] = -1

    foreground = np.sum(roi_anchor[:, 0] == 1)
    return roi_anchor, foreground

def compute_overlap(mat1, mat2):
    s1 = mat1.shape[0]
    s2 = mat2.shape[0]
    area1 = (mat1[:, 2] - mat1[:, 0]) * (mat1[:, 3] - mat1[:, 1])
    if mat2.shape[1] == 5:
        area2 = mat2[:, 4]
    else:
        area2 = (mat2[:, 2] - mat2[:, 0]) * (mat2[:, 3] - mat2[:, 1])

    x1 = cartesian([mat1[:, 0], mat2[:, 0]])

    x1 = np.amax(x1, axis=1)
    x2 = cartesian([mat1[:, 2], mat2[:, 2]])
    x2 = np.amin(x2, axis=1)
    com_zero = np.zeros(x2.shape[0])
    w = x2 - x1
    w = w - 1

    w = np.maximum(com_zero, w)

    y1 = cartesian([mat1[:, 1], mat2[:, 1]])
    y1 = np.amax(y1, axis=1)
    y2 = cartesian([mat1[:, 3], mat2[:, 3]])
    y2 = np.amin(y2, axis=1)
    h = y2 - y1
    h = h - 1
    h = np.maximum(com_zero, h)

    oo = w * h

    aa = cartesian([area1[:], area2[:]])
    aa = np.sum(aa, axis=1)

    ooo = oo / (aa - oo)

    overlap = np.transpose(ooo.reshape(s1, s2), (1, 0))

    return overlap

def compute_regression(mat1, mat2):
    target = np.zeros(4)
    w1 = mat1[2] - mat1[0]
    h1 = mat1[3] - mat1[1]
    w2 = mat2[2] - mat2[0]
    h2 = mat2[3] - mat2[1]

    target[0] = (mat1[0] - mat2[0]) / w2
    target[1] = (mat1[1] - mat2[1]) / h2
    target[2] = np.log(w1 / w2)
    target[3] = np.log(h1 / h2)

    return target
