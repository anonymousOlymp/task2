from math import degrees, radians
import cv2

def main() -> None:
    try:
        image = None
        while image is None:
            open_file_name = input("Enter the name of the file to open: ")
            image = cv2.imread(open_file_name)
            if (image is None):
                print("Error. File can't be read!")
        width, height = image.shape[:2]
        angle = find_angle(image)
        rotated_image = rotate(image, width, height, angle)
        was_written = False
        while not was_written:
            save_file_name = input("Enter the name of the file to save: ")
            was_written = cv2.imwrite(save_file_name, rotated_image)
            if not was_written:
                print("Error. File can't be written!")
    except Exception as e:
        print(f"Error. Unexpected exception {e} was caught!")


def find_angle(image):
    grayscaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    NORMALIZE_MINIMUM = 0
    NORMALIZE_MAXIMUM = 255
    grayscaled_image = cv2.normalize(grayscaled_image, None, alpha=NORMALIZE_MINIMUM, beta=NORMALIZE_MAXIMUM, norm_type=cv2.NORM_MINMAX)
    LOWER_THRESHOLD = 64
    UPPER_THRESHOLD = 128
    edges = cv2.Canny(grayscaled_image, LOWER_THRESHOLD, UPPER_THRESHOLD, apertureSize = 3)
    angle = find_lines_angle(edges)
    return degrees(angle) - 90

def rotate(image, width, height, angle):
    center = (height / 2., width / 2.)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, matrix, (width, height))
    return rotated

def find_lines_angle(edges) -> int:
    avg_theta = 0.
    min_threshold = 1
    max_threshold = 255
    MSE_EPSILON = radians(5) ** 2
    while max_threshold - min_threshold > 1:
        threshold = (min_threshold + max_threshold) // 2
        lines = cv2.HoughLines(edges, 1, radians(1), threshold)
        if lines is None:
            min_threshold = threshold
            continue
        num_thetas = len(lines)
        mse_thetas = 0.
        for line in lines:
            _, theta = line[0]
            avg_theta += (theta / num_thetas)
            mse_thetas += (theta ** 2 / num_thetas)
        mse_thetas -= (avg_theta ** 2)
        if mse_thetas > MSE_EPSILON:
            max_threshold = threshold
            continue
        if avg_theta >= radians(30) and avg_theta <= radians(150):
            raise RuntimeError("Error. Direction can't be detected!")
        return avg_theta
    raise RuntimeError("Error. Required accuracy can't be reached!")

if __name__ == "__main__":
    main()
