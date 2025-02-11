from math import degrees, radians
import cv2
import numpy as np


def solve():
    image = cv2.imread('./img/origin.png')
    assert image is not None, "file could not be read, check with os.path.exists()"
    width, height = image.shape[:2]
    grayscaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscaled_image, 50, 150, apertureSize = 3)
    lines = cv2.HoughLines(edges, 1, radians(1), 100)
    assert lines is not None
    avg_theta = 0.
    num_thetas = len(lines)
    for line in lines:
        rho,theta = line[0]
        avg_theta += (theta / num_thetas)
    
    assert avg_theta >= np.pi / 6 and avg_theta <= 5 * np.pi / 6
    M = cv2.getRotationMatrix2D(((height-1)/2.0,(width-1)/2.0),degrees(avg_theta) - 90,1)
    rotated = cv2.warpAffine(image, M, (width, height))
    cv2.imwrite('./img/rotated.png', rotated)

def main() -> None:
    solve()

if __name__ == "__main__":
    main()
