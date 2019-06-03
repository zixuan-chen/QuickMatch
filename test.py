from quickmatch_cluster import *
import cv2


def get_size(path):
    size = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            size[line[0]] = [int(line[1]), int(line[2])]
    return size


data_path = "C:\\Users\\ChenZixuan\\Documents\\Graffiti_dataset"
categroy = 'graf'
size = get_size(data_path + "\\size")
row, col = size[categroy]

f = np.load(data_path + '\\%s\\%s.npz'%(categroy, categroy))
kp, des = f['kp'], f['des']
n, m, t = des.shape
# n is number of points per image
# m is number of image
data = des.transpose(1, 0, 2).reshape(m*n, t)
n, m, t = kp.shape
kp = kp.transpose(1, 0, 2).reshape(m*n, t)

clusters = quickmatch_cluster(data, n, kernel=np.exp, flag_low_memory=False)

for i in range(6):
    cv2.namedWindow("imgs[%d]" % (i+1), cv2.WINDOW_NORMAL)

# for i in range(kp.shape[0]):
#
#     for j in range(6):
#         cv2.circle(imgs[j], (kp[i][j][0], kp[i][j][1]), 2, (0,255,255),3)
#         cv2.imshow("imgs[%d]"%(j+1), imgs[j])
#
#     cv2.waitKey(0)
imgs = np.zeros((6, col, row, 3), dtype=np.uint8)

for c in clusters:
    for i in range(6):
        img_path = data_path + "\\%s\\img%d.jpg" % (categroy, i+1)
        imgs[i] = cv2.imread(img_path, cv2.IMREAD_COLOR)

    for idx in c:
        i = idx // n
        x, y = kp[idx][0], kp[idx][1]
        cv2.circle(imgs[i], (x, y), 2, (0, 255, 255), 3)

    for i in range(6):
        cv2.imshow("imgs[%d]" % (i + 1), imgs[i])
    cv2.waitKey(0)