import numpy as np

#  Felzenszwalb et al.
def non_max_suppression_slow(boxes, overlapThresh):
  # if there are no boxes, return an empty list
  if len(boxes) == 0:
    return boxes

  # initialize the list of picked indexes
  pick = []

  # grab the coordinates of the bounding boxes
  x1 = boxes[:,0]
  y1 = boxes[:,1]
  xw = boxes[:,2]
  yh = boxes[:,3]
  x2 = x1 + xw
  y2 = y1 + yh

  # compute the area of the bounding boxes and sort the bounding
  # boxes by the bottom-right y-coordinate of the bounding box
  area = xw * yh
  idxs = np.argsort(boxes[:,4])

  # keep looping while some indexes still remain in the indexes
  # list
  while len(idxs) > 0:
    # grab the last index in the indexes list, add the index
    # value to the list of picked indexes, then initialize
    # the suppression list (i.e. indexes that will be deleted)
    # using the last index
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)
    suppress = [last]

    # loop over all indexes in the indexes list
    for pos in xrange(0, last):
      # grab the current index
      j = idxs[pos]

      # find the largest (x, y) coordinates for the start of
      # the bounding box and the smallest (x, y) coordinates
      # for the end of the bounding box
      xx1 = max(x1[i], x1[j])
      yy1 = max(y1[i], y1[j])
      xx2 = min(x2[i], x2[j])
      yy2 = min(y2[i], y2[j])

      # compute the width and height of the bounding box
      w = max(0, xx2 - xx1)
      h = max(0, yy2 - yy1)

      # compute the ratio of overlap between the computed
      # bounding box and the bounding box in the area list
      overlap = float(w * h) / min(area[j], area[i])

      # if there is sufficient overlap, suppress the
      # current bounding box
      if overlap > overlapThresh:
        suppress.append(pos)

    # delete all indexes from the index list that are in the
    # suppression list
    idxs = np.delete(idxs, suppress)

  # return only the bounding boxes that were picked
  return boxes[pick]

def non_max_suppression_fast(boxes, overlapThresh):
  # if there are no boxes, return an empty list
  if len(boxes) == 0:
    return boxes
 
  # initialize the list of picked indexes 
  pick = []
 
  # grab the coordinates of the bounding boxes
  x1 = boxes[:,0]
  y1 = boxes[:,1]
  xw = boxes[:,2]
  yh = boxes[:,3]
  x2 = x1 + xw
  y2 = y1 + yh
 
  # compute the area of the bounding boxes and sort the bounding
  # boxes by the bottom-right y-coordinate of the bounding box
  area = xw * yh
  idxs = np.argsort(boxes[:,4])
 
  # keep looping while some indexes still remain in the indexes
  # list
  while len(idxs) > 0:
    # grab the last index in the indexes list and add the
    # index value to the list of picked indexes
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)
 
    # find the largest (x, y) coordinates for the start of
    # the bounding box and the smallest (x, y) coordinates
    # for the end of the bounding box
    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.minimum(x2[i], x2[idxs[:last]])
    yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
 
    # compute the ratio of overlap
    overlap = (w * h) / np.minimum(area[idxs[:last]], area[i])
 
    # delete all indexes from the index list that have
    idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
  return boxes[pick]

def filter_bbox(bbox):
  xrng = [0.1, 959.9]
  yrng = [0.1, 719.9]
  #bbox[:, :4] = bbox[:, :4] / 1.5
  x1 = bbox[:, 0]
  y1 = bbox[:, 1]
  x2 = bbox[:, 0] + bbox[:, 2]
  y2 = bbox[:, 1] + bbox[:, 3]
  keep = np.where((x1 > xrng[0]) & (x2 < xrng[1]) & (y1 > yrng[0]) & (y2 < yrng[1]))[0]
  return bbox[keep, :]