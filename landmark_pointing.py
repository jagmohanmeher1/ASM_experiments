import cv2
import numpy as np
import os

# Initialize variables
landmarks = []
image = None

# Mouse callback function to store landmark points
def select_landmark(event, x, y, flags, param):
    global landmarks, image

    if event == cv2.EVENT_LBUTTONDOWN:
        landmarks.append((x, y))
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        cv2.putText(image, str(len(landmarks)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.imshow('Image with Landmarks', image)

def resize_image_to_screen(image, screen_width, screen_height):
    image_height, image_width = image.shape[:2]
    aspect_ratio = float(image_width) / float(image_height)

    if aspect_ratio > 1:
        new_width = min(image_width, screen_width)
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = min(image_height, screen_height)
        new_width = int(new_height * aspect_ratio)

    return cv2.resize(image, (new_width, new_height))

def main():
    global image

    # Load image
    image_path = '/Users/jagmohanmeher/Documents/NCKU/4th sem/ASM experiments/jag.jpg'
    image_name = os.path.basename(image_path)
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image")
        return

    # Resize image to fit screen
    screen_width, screen_height = 1200, 800  # Change these values according to your screen size
    image = resize_image_to_screen(image, screen_width, screen_height)

    # Create a window and display the image
    cv2.namedWindow('Image with Landmarks')
    cv2.setMouseCallback('Image with Landmarks', select_landmark)
    cv2.imshow('Image with Landmarks', image)

    # Wait for the 'q' key to be pressed to exit
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save landmarks to a file
    with open('landmarks.txt', 'w') as f:
        f.write(f'Image Name: {image_name}\n')
        for point in landmarks:
            f.write(f'{point[0]}, {point[1]}\n')

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
