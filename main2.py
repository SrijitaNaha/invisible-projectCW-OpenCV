import cv2
import numpy as np
import time


def initialize_video_capture(video_path):
    """Initialize video capture from the specified file."""
    return cv2.VideoCapture(video_path)


def capture_background(video_capture, frames=60):
    """Capture the background frame by reading the first few frames."""
    background = None
    for _ in range(frames):
        return_val, frame = video_capture.read()
        if return_val:
            background = frame
    return np.flip(background, axis=1) if background is not None else None


def create_mask(hsv_frame):
    """Create a mask for red color in the HSV frame."""
    lower_red1 = np.array([100, 40, 40])
    upper_red1 = np.array([100, 255, 255])
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)

    lower_red2 = np.array([155, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)

    mask = mask1 + mask2
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2
    )
    return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)


def process_frame(frame, background):
    """Process a single frame to create the invisible effect."""
    flipped_frame = np.flip(frame, axis=1)
    hsv_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2HSV)

    mask = create_mask(hsv_frame)
    mask_inverse = cv2.bitwise_not(mask)

    res1 = cv2.bitwise_and(background, background, mask=mask)
    res2 = cv2.bitwise_and(flipped_frame, flipped_frame, mask=mask_inverse)

    return cv2.addWeighted(res1, 1, res2, 1, 0)


def main(video_path):
    """Main function to run the invisible man effect."""
    raw_video = initialize_video_capture(video_path)

    time.sleep(1)  # Allow time for the camera to warm up
    background = capture_background(raw_video)

    if background is None:
        print("Failed to capture background.")
        return

    while raw_video.isOpened():
        return_val, img = raw_video.read()
        if not return_val:
            break

        final_output = process_frame(img, background)
        cv2.imshow("INVISIBLE MAN", final_output)

        if cv2.waitKey(10) == 27:  # Exit on ESC key
            break

    raw_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("cat.mp4")
