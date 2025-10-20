import numpy as np
import cv2
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt

def get_gray_frame(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def initialise_particles(N,cov_matrix, center_pos):
    #Initialisation of points and weights
    noise = np.random.multivariate_normal(np.zeros(3), cov_matrix, N)
    particles = center_pos + noise
    weights = np.ones(N) / N

    return particles, weights


def point_square_rotation(center_pos):
    x = center_pos[0]
    y = center_pos[1]
    alpha = np.pi * center_pos[2] / 180  # conversion degrés → radians

    # Ordre Bas_gauche/Bas-Droit/Haut-droit/haut-gauche
    square_corner_before_rotation = [
        [x - 20, y - 20],
        [x + 20, y - 20],
        [x + 20, y + 20],
        [x - 20, y + 20]
    ]

    square_corner_after_rotation = []

    # Rotation
    for coords in square_corner_before_rotation:
        xr = x + (coords[0] - x) * np.cos(alpha) - (coords[1] - y) * np.sin(alpha)
        yr = y + (coords[0] - x) * np.sin(alpha) + (coords[1] - y) * np.cos(alpha)
        square_corner_after_rotation.append([xr, yr])

    # Bornes du grand carré
    min_x = min(pt[0] for pt in square_corner_after_rotation)
    max_x = max(pt[0] for pt in square_corner_after_rotation)
    min_y = min(pt[1] for pt in square_corner_after_rotation)
    max_y = max(pt[1] for pt in square_corner_after_rotation)

    print("min_x :", min_x)
    print("max_x :", max_x)
    print("min_y :", min_y)
    print("max_y :", max_y)
    print("Avant rotation :", square_corner_before_rotation)
    print("Après rotation :", square_corner_after_rotation)

    # Coins du grand carré englobant
    big_square_corner = [
        [min_x, max_y],
        [max_x, max_y],
        [min_x, min_y],
        [max_x, min_y]
    ]

    # Coordonnées discrètes à l’intérieur du carré englobant
    x_range = np.arange(round(min_x), round(max_x) + 1)
    y_range = np.arange(round(min_y), round(max_y) + 1)
    big_square_points = (x_range, y_range)

    return square_corner_after_rotation, big_square_corner, big_square_points


def pixels_in_rotated_square(square_corner_after_rotation, big_square_pixels):
    #return the list of pixel inside the rotated square

    x_range, y_range = big_square_pixels
    corners = np.array(square_corner_after_rotation)
    inside_pixels = []

    # Parcours des pixels du carré englobant
    for x in x_range:
        for y in y_range:
            P = np.array([x, y])
            inside = True

            # Test de positionnement par produit vectoriel (règle de la main droite)
            for i in range(4):
                A = corners[i]
                B = corners[(i + 1) % 4]
                AB = B - A
                AP = P - A

                # Produit vectoriel 2D (composante z)
                cross_z = AB[0] * AP[1] - AB[1] * AP[0]

                if i == 0:
                    ref_sign = np.sign(cross_z)
                else:
                    if np.sign(cross_z) != ref_sign and np.sign(cross_z) != 0:
                        inside = False
                        break

            if inside:
                inside_pixels.append((x, y))

    return inside_pixels

def define_buffer(particle, width, heigth, frame):
    x_min = int(max(particle[1] - width, 0))
    x_max = int(min(particle[1] + width, frame.shape[1]))
    y_min = int(max(particle[0] - heigth, 0))
    y_max = int(min(particle[0] + heigth, frame.shape[0]))

    center_rectangle = frame[y_min:y_max, x_min:x_max]

    #Rajouter si on a le temps pour la rotation

    return center_rectangle


def update_weights(frame, object_tracked, listparticles, list_weights, hist_ref):
    # Update particle weights based on their similarity to the reference histogram
    lambda_ = 20
    update_weights = np.zeros(len(listparticles))
    for i, particle in enumerate(listparticles):

        height, width = object_tracked.shape[:2]

        object_buffer = define_buffer(particle, width, height, frame)

        hist, bin_edges = np.histogram(object_buffer, bins=128, range=(0, 127), density=True)
        dist = np.sqrt(1 - np.sum(np.sqrt(hist * hist_ref)))
        update_weights[i] = np.exp(-lambda_ * dist ** 2)*list_weights[i]
    update_weights /= np.sum(update_weights)  # Normalize weights
    return update_weights


def predict_particles(particles, cov_matrix):
    #Set new position of each particle
    for i in range(len(particles)) :
        particles[i] += np.random.multivariate_normal(np.array([0,0,0]),cov_matrix)
    return particles

def estimate_position(list_particles, list_weights):
    # Estimate the object's position (weighted average)
    return np.average(list_particles, weights=list_weights, axis=0)

def multinomial_resampling(list_particles, weights):
    # multinomial resampling
    N = len(list_particles)
    indices = np.random.choice(N, size=N, replace=True, p=weights)
    new_particles = list_particles[indices]
    new_weights = np.ones(N) / N

    return new_particles, new_weights


def main():
    covariance_matrix = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]])

    file_path = "/Users/antoine/Documents/Cours/3A/TIV/TP2/video sequences/synthetic/escrime-4-3.avi"

    # Importation of the video
    cap = cv2.VideoCapture(file_path)
    ret, frame = cap.read()

    gray_frame = get_gray_frame(frame)

    # Initialisation in the center
    center_rectangle = gray_frame[220:260, 300:340]
    center_gray = gray_frame[240, 320]
    center_pos = [240, 320, 0]

    N = 15 # number of particles

    # Compute the histogram
    hist_ref, bin_edges = np.histogram(center_rectangle, bins=128, range=(0, 127), density=True)

    list_particles, weights = initialise_particles(N, covariance_matrix, center_pos)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = get_gray_frame(frame)

        # Predict new particle positions
        list_particles = predict_particles(list_particles, covariance_matrix)

        # Update weights based on histogram similarity
        weights = update_weights(gray_frame, list_particles, weights, hist_ref)

        # Resample particles
        list_particles, weights = multinomial_resampling(list_particles, weights)

        # Estimate the new object position
        estimated_position = estimate_position(list_particles, weights)


        center = (estimated_position[1], estimated_position[0])
        size = (40, 40)
        angle = np.degrees(estimated_position[2])

        # Crée le rectangle et récupère ses coins
        rect = ((center[0], center[1]), size, angle)
        pts = cv2.boxPoints(rect).astype(int)

        # Dessine le carré
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow('Tracking', frame)
        if cv2.waitKey(30) & 0xFF == 27:  # Esc pour quitter
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()






















        
