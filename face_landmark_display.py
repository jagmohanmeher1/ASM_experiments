import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

def read_landmarks_from_file(file_path):
    images_landmarks = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.strip().split()
            image_name = tokens[0]
            landmarks = np.array([float(x) for x in tokens[1:]]).reshape(-1, 2)
            images_landmarks[image_name] = landmarks
    return images_landmarks

def display_image_with_landmarks(image_path, landmarks):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image")
        return

    for idx, point in enumerate(landmarks):
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        # cv2.putText(image, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.namedWindow('Image with Landmarks')
    cv2.imshow('Image with Landmarks', image)

    # Wait for the 'q' key to be pressed to exit
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    
def plot_image_with_landmarks(image, landmarks):
    image_copy = image.copy()
    for idx, point in enumerate(landmarks):
        x, y = int(point[0]), int(point[1])
        cv2.circle(image_copy, (x, y), 2, (0, 0, 255), -1)
        #cv2.putText(image_copy, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

###Function for performing Procrustes analysis
def generalized_procrustes_analysis(landmarks_list, max_iterations=100, tolerance=1e-10):
    mean_shape = np.mean(landmarks_list, axis=0)

    for iteration in range(max_iterations):
        aligned_landmarks_list = []

        for landmarks in landmarks_list:
            _, aligned_landmarks, _ = procrustes(mean_shape, landmarks)
            aligned_landmarks_list.append(aligned_landmarks)

        new_mean_shape = np.mean(aligned_landmarks_list, axis=0)
        mean_shape_change = np.linalg.norm(new_mean_shape - mean_shape)

        mean_shape = new_mean_shape

        if mean_shape_change <= tolerance:
            break

    return mean_shape, aligned_landmarks_list

def main():
    landmarks_file_path = '/Users/jagmohanmeher/Documents/NCKU/4th sem/ASM experiments/FRGC/frgc_train.txt'
    images_folder = '/Users/jagmohanmeher/Documents/NCKU/4th sem/ASM experiments/FRGC'
    images_landmarks = read_landmarks_from_file(landmarks_file_path)

    ### Display the first image with landmarks
    # first_image_name, first_image_landmarks = list(images_landmarks.items())[0]
    # first_image_path = os.path.join(images_folder, first_image_name)
    # display_image_with_landmarks(first_image_path, first_image_landmarks)
    
    ###Display first 5 images on the plot to compare 
    # fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    # for idx, (image_name, landmarks) in enumerate(list(images_landmarks.items())[:5]):
    #     image_path = os.path.join(images_folder, image_name)
    #     image = cv2.imread(image_path)
    #     if image is None:
    #         print(f"Error: Could not load image {image_name}")
    #         continue

    #     image_with_landmarks = plot_image_with_landmarks(image, landmarks)
    #     axes[idx].imshow(image_with_landmarks)
    #     axes[idx].set_title(image_name)
    #     axes[idx].axis('off')

    # plt.show()
    
    ###Plot all the landmarks as a point cloud.
    # all_landmarks = []

    # for image_name, landmarks in images_landmarks.items():
    #     all_landmarks.extend(landmarks)

    # all_landmarks = np.array(all_landmarks)

    # plt.figure(figsize=(10, 10))
    # plt.scatter(all_landmarks[:, 0], all_landmarks[:, 1], s=3, c='r')
    # plt.title('Point Cloud of All Landmarks')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.gca().invert_yaxis()
    # plt.show()

    ###Point cloud of only first 10 images
    # all_landmarks = []

    # for idx, (image_name, landmarks) in enumerate(images_landmarks.items()):
    #     if idx >=1:
    #         break
    #     all_landmarks.extend(landmarks)

    # all_landmarks = np.array(all_landmarks)

    # plt.figure(figsize=(10, 10))
    # plt.scatter(all_landmarks[:, 0], all_landmarks[:, 1], s=5, c='r')
    # plt.title('Point Cloud of First 10 Images Landmarks')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.gca().invert_yaxis()
    # plt.show()
    
    ###Perform PCA on all the landmarks and plot the mean shape
    all_landmarks = []

    for idx, (image_name, landmarks) in enumerate(images_landmarks.items()):
        all_landmarks.append(np.array(landmarks))

    mean_shape, aligned_landmarks_list = generalized_procrustes_analysis(all_landmarks)

    pca = PCA(n_components=2)
    pca.fit(np.array(aligned_landmarks_list).reshape(len(aligned_landmarks_list), -1))

    mean_face = pca.mean_.reshape(mean_shape.shape)

    plt.figure(figsize=(10, 10))
    plt.scatter(mean_face[:, 0], mean_face[:, 1], s=50, c='r', marker='o')
    plt.title('Mean Face Model')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()
    plt.show()
    
if __name__ == '__main__':
    main()
