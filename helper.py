import numpy as np
import cv2

def draw_polygon( img, points, color = (255, 0, 0) ):
    pts = points.astype(int)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, color, 1 )


if __name__ == "__main__":
    img = cv2.imread('./test_images/straight_lines1.jpg')
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
         [((img_size[0] / 6) - 10), img_size[1]],
         [(img_size[0] * 5 / 6) + 60, img_size[1]],
         [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    draw_polygon( img, src )

    cv2.imshow('test', img)
    cv2.waitKey(0)