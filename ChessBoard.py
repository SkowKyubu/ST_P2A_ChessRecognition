import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import Params
import itertools


def get_chess_board(img):
    # On récupère les dimensions de l'image
    img_shape = img.shape

    # On transforme l'image en niveau de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detecter les contours avec l'algorithme de Canny
    edges = cv2.Canny(gray, 100, 200)

    # Appliquer la transformée de Hough sur les contours détectés
    lines = cv2.HoughLines(edges, 1, np.pi/720.0, 50, np.array([]), 0, 0)

    # On ne sélectionne que 200 lignes parmi les lignes détectées
    lines_best = np.squeeze(lines)[:Params.line_amount]

    # On calcule les coordonnées des intersections entre les lignes
    intersections, intersections_info, parallel_sets_list = get_all_line_intersection(lines_best, img_shape)

    # On récupère les clusters de ligne : c.a.d les ensemble de lignes impliquée dans une même intersection
    intersecting_clusters = get_intersecting_line_clusters(intersections, intersections_info)

    # On récupère les indices des paires de droites parallèles regroupées en clusters
    parallel_clusters = get_parallel_line_clusters(lines_best, parallel_sets_list)

    # On renvoie les deux ensemble principaux de droites parallèles
    # Relatives aux droites des lignes de fuite de l'échiquier
    best_cluster_pair = select_best_performing_cluster_pair(lines_best, intersecting_clusters, parallel_clusters)

    # On récupère les droites relatives aux lignes de fuites.
    cluster_means = [cluster_mean_hessfixed(lines_best, best_cluster_pair[0]),
                     cluster_mean_hessfixed(lines_best, best_cluster_pair[1])]

    # On élimine les clusters de lignes
    best_cluster_pair_duplicate_eliminated = [
        cluster_eliminate_duplicate(lines_best, best_cluster_pair[0], cluster_means[1], img.shape),
        cluster_eliminate_duplicate(lines_best, best_cluster_pair[1], cluster_means[0], img.shape)]

    # On élimite les lignes n'étant pas relatives à l'échiquier
    best_cluster_pair_chessboard = cluster_eliminate_non_chessboard(best_cluster_pair_duplicate_eliminated,
                                                                               cluster_means, img.shape)

    # On calcule les sommets de chaque cases
    all_corners_in_chessboard = get_intersections_between_clusters(best_cluster_pair_chessboard[0],
                                                                              best_cluster_pair_chessboard[1],
                                                                              img.shape)

    # Pour chaque case on cree une liste contenant les quatres sommets de la case
    coordinate_case = get_coordinate_boxes(all_corners_in_chessboard)
    return all_corners_in_chessboard, coordinate_case


def draw_chess_board(img, all_corners_in_chessboard):
    color = (0, 0, 255)
    # Tracé des lignes verticales
    for k in all_corners_in_chessboard:
        x1, y1 = k[0]
        x2, y2 = k[len(k) - 1]
        cv2.line(img, (x1, y1), (x2, y2), color, 2)
    # Tracé des lignes horizonales
    for i in range(9):
        x1, y1 = all_corners_in_chessboard[0][i]
        x2, y2 = all_corners_in_chessboard[8][i]
        cv2.line(img, (x1, y1), (x2, y2), color, 2)

    # Tracer les intersections sur l'image
    for ligne_chess in all_corners_in_chessboard:
        for case in ligne_chess:
            cv2.circle(img, case, 5, (0, 255, 0), -1)
    return img

def intersection(line1, line2, img_shape):
    # On récupère les coordonnées polaires des lignes 1 et 2 :
    rho1, theta1 = line1
    rho2, theta2 = line2

    # On construit la matrice A et le vecteur b relative au système d'équation
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    try:
        # On résout le système d'équation linéaire Ax+b pour trouver le point d'intersection
        x0, y0 = np.linalg.solve(A, b)
    except:
        # Dans le cas où la résolution échoue on renvoie les coordonnées (-1,-1)
        x0, y0 = (-1, -1)
    x0, y0 = int(np.round(x0)), int(np.round(y0))

    # On vérifie si les intersections sont en dehors des limites de l'image dans quel cas on renvoie (-1,1)
    if abs(y0) > img_shape[0] * Params.parallel_threshold or abs(x0) > img_shape[0] * Params.parallel_threshold:
        x0, y0 = (-1, -1)
    return [x0, y0]


def split_clusters_using_labels(all_clusters, labels):
    cluster_list = []
    # On crée pour chaque cluster, on crée un ensemble de toutes les droites impliquées dans le cluster
    for cluster_id in range(max(labels) + 1):
        mask = (labels == cluster_id)
        # On cree un ens des paires de droite impliquée dans le cluster
        cluster_list.append(np.array(all_clusters)[mask])

    return cluster_list


def fix_hessian_form(line, reverse=False):
    if not reverse and line[0] < 0:
        new_rho = - (line[0])
        new_alpha = -(np.pi - line[1])
        return new_rho, new_alpha
    elif reverse and line[1] < 0:
        new_rho = - (line[0])
        new_alpha = np.pi + line[1]
        return new_rho, new_alpha
    return line


def fix_hessian_form_vectorized(lines):
    lines = lines.copy()
    neg_rho_mask = lines[:, 0] < 0
    # On s'assure que les valeurs de rho soient positives
    lines[neg_rho_mask, 0] *= -1
    #On s'assure que toutes les lignes soient orientées dans la même orientation
    lines[neg_rho_mask, 1] -= np.pi
    return lines


def angle_diff(line1, line2):
    diff = float('inf')
    if (line1[0] < 0) ^ (line2[0] < 0):
        if line1[0] < 0:
            diff = abs(fix_hessian_form(line1)[1] - line2[1]) % (np.pi)
        else:
            diff = abs(line1[1] - fix_hessian_form(line2)[1]) % (np.pi)

    diff = min(diff, abs(line1[1] - line2[1]) % np.pi)

    return diff


def angle_diff_vectorized(lines, line_to_calculate_diff):
    hess_fixed_lines = fix_hessian_form_vectorized(lines)
    hess_fixed_calculate_line = fix_hessian_form(line_to_calculate_diff)

    diff_fixed = np.full(lines.shape[0], float('inf'))
    hess_test_mask = lines[:, 0] < 0

    if line_to_calculate_diff[0] >= 0:
        diff_fixed[hess_test_mask] = np.mod(np.abs(hess_fixed_lines[hess_test_mask, 1] - line_to_calculate_diff[1]),
                                            np.pi)
    else:
        diff_fixed[~hess_test_mask] = np.mod(lines[~hess_test_mask, 1] - hess_fixed_calculate_line[1], np.pi)

    diff_normal = np.mod(np.abs(lines[:, 1] - line_to_calculate_diff[1]), np.pi)

    return np.minimum(diff_normal, diff_fixed)


def cluster_mean_hessfixed(lines, cluster):
    cluster_lines = lines[list(cluster)]
    normal_mean = np.mean(cluster_lines, axis=0)

    # On corrige les lignes calculées pour s'assurer qu'elles soient toutes représentées de la même manière
    hess_fixed_cluster_lines = fix_hessian_form_vectorized(cluster_lines)

    #On calcule l'orientation moyenne des droites des clusters
    hess_fixed_mean = np.mean(hess_fixed_cluster_lines, axis=0)

    normal_mean_diff = np.mean(angle_diff_vectorized(cluster_lines, normal_mean))
    hess_fixed_mean_diff = np.mean(angle_diff_vectorized(cluster_lines, hess_fixed_mean))

    return normal_mean if normal_mean_diff < hess_fixed_mean_diff else fix_hessian_form(hess_fixed_mean, True)


def get_all_line_intersection(lines, img_shape):
    parallel_sets_list = list() # Permet de stocker les positions des intersections
    intersections_info = list() # Permet de stocker les indices des paires de droite de l'intersection
    intersections = list() # Permet de strocker les groupes de droites parallèles
    for i, line in enumerate(lines):
        for j, line in enumerate(lines[i:], start=i):
            if i == j:
                continue
            # On calcule la position des intersections
            line_intersection = intersection(lines[i], lines[j], img_shape)
            # Si les coordonnées renvoyée par intersection() sont égales à '1, les droites ne sont pas sécantes
            if line_intersection[0] == -1 and line_intersection[1] == -1:
                set_exists = False
                # On vérifie si ces droites parallèles appartiennent à l'ensemble parallel_sets_list
                for next_set in parallel_sets_list:
                    # Dans le cas où les lignes i ou j apparitiennent à un ens de lignes parallèles, on ajoute i et j à l'ens
                    if (i in next_set) or (j in next_set):
                        set_exists = True
                        next_set.add(i) # ajoute i à l'ensemble
                        next_set.add(j) # ajoute j à l'ensemble
                        break
                # Sinon on cree un nouvel ensemble de lignes parallèles
                if not set_exists:
                    parallel_sets_list.append({i, j})
            else:
                # Si les droites sont sécantes, on vérifie que l'intersection se trouve dans l'image
                # On ignore les intersections situées dans l'image
                if not ((0 < line_intersection[0] < img_shape[0]) and (0 < line_intersection[1] < img_shape[1])):
                    intersections_info.append((i, j))
                    intersections.append(line_intersection)
    return intersections, intersections_info, sorted(parallel_sets_list, key=len, reverse=True)


def get_intersecting_line_clusters(intersections, intersections_info):
    # On applique DBSCAN afin de considéré uniquement les cluster de ligne
    dbscan_intersections = DBSCAN(eps=Params.dbscan_eps_intersection_clustering, min_samples=8).fit(intersections) # 10, 8
    labels_intersections = dbscan_intersections.labels_

    # On récupère chaque ensemble de cluster (ensemble de paires de droite liée à un cluster)
    intersection_clusters = split_clusters_using_labels(intersections_info, labels_intersections)

    # On élimite les doublons de ligne dans le cluster
    unique_lines_each_cluster = list()
    for cluster in intersection_clusters:
        unique_lines = set()
        for lines in cluster:
            unique_lines.add(lines[0])
            unique_lines.add(lines[1])
        unique_lines_each_cluster.append(unique_lines)
    return sorted(unique_lines_each_cluster, key=len, reverse=True)


def get_parallel_line_clusters(lines, parallel_sets):
    cur_sets = parallel_sets
    cur_means = list()
    # On calcule la moyenne de la coordonnée y de chaque ensemble de droites parallèles
    for next_set in cur_sets:
        cur_means.append(np.mean(lines[list(next_set)], axis=0)[1])

    i = 0
    while i < (len(cur_sets) - 1):
        for j in range(i + 1, len(cur_sets)):
            # Si la différence abs entre les moyennes de coordonnées y est inf à un seuil
            if abs(cur_means[i] - cur_means[j]) < Params.parallel_angle_threshold:
                # On fusionne les ensembles
                cur_sets[i] = cur_sets[i] | cur_sets[j]
                # Puis on recalcule la moyenne de coordonnées y
                cur_means[i] = np.mean(lines[list(cur_sets[i])], axis=0)[1]
                cur_sets.pop(j)
                cur_means.pop(j)
                i = 0
                break
        i += 1
    return sorted(cur_sets, key=len, reverse=True)


def select_best_performing_cluster_pair(lines, intersections, parallel_sets):
    # Fusion des ensembles de droites qui se croisent avec ceux qui sont parallèles
    merged_clusters = intersections + parallel_sets
    # Calcul de la taille de chaque ensemble
    merged_sizes = list(map(lambda x: len(x), merged_clusters))

    pass_list = list()
    for i, cluster_i in enumerate(merged_clusters):
        for j, cluster_j in enumerate(merged_clusters[i:], start=i):
            if i == j:
                continue
            if angle_diff(cluster_mean_hessfixed(lines, cluster_i), cluster_mean_hessfixed(lines, cluster_j)) > Params.two_line_cluster_threshold:
                pass_list.append((i, j))

    pass_list.sort(key = lambda x: (merged_sizes[x[0]] * merged_sizes[x[1]]), reverse=True)
    winner_pair = pass_list[0]

    return merged_clusters[winner_pair[0]], merged_clusters[winner_pair[1]]


def cluster_eliminate_duplicate(lines, cluster, intersect_line, img_shape):
    cluster_lines = lines[list(cluster)]
    intersection_points = list(map(lambda x: intersection(x, intersect_line, img_shape), cluster_lines))

    dbscan_test = DBSCAN(eps=Params.dbscan_eps_duplicate_elimination, min_samples=1).fit(
        intersection_points)
    labels_test = dbscan_test.labels_

    merged_cluster = list()
    for i in range(max(labels_test) + 1):
        mask = (labels_test == i)
        merged_cluster.append(cluster_mean_hessfixed(lines, np.array(list(cluster))[mask]))

    return merged_cluster


def cluster_eliminate_non_chessboard(merged_clusters, cluster_means, img_shape):
    first_cluster, second_cluster = merged_clusters

    mean_first_cluster, mean_second_cluster = cluster_means

    intersections_first_cluster = list(map(lambda x: intersection(x, mean_second_cluster, img_shape), first_cluster))

    intersections_second_cluster = list(map(lambda x: intersection(x, mean_first_cluster, img_shape), second_cluster))

    best_intersections_first_cluster = select_nine_fittable_intersections(intersections_first_cluster)
    best_intersections_second_cluster = select_nine_fittable_intersections(intersections_second_cluster)

    return (np.array(first_cluster)[best_intersections_first_cluster],
            np.array(second_cluster)[best_intersections_second_cluster])


def select_nine_fittable_intersections(intersections):
    np_intersections = np.array(intersections)

    axis_variance = np.var(np_intersections, axis=0)

    metric_col = (0 if axis_variance[0] > axis_variance[1] else 1)
    metric_value = np_intersections[:, metric_col]

    sorted_idx = np.argsort(metric_value)
    metric_value = metric_value[sorted_idx]

    all_combinations_iter = itertools.combinations(np.array(list(enumerate(metric_value))), 9)
    all_combinations = np.stack(np.fromiter(all_combinations_iter, dtype=tuple))

    x = range(9)
    fitter = lambda y: np.poly1d(np.polyfit(x, y, Params.polynomial_degree))(x)
    all_combinations_fitted_calculated = np.apply_along_axis(fitter, 1, all_combinations[:, :, 1])

    all_combinations_mse = (np.square(all_combinations[:, :, 1] - all_combinations_fitted_calculated)).mean(axis=1)

    best_combination_indexes = all_combinations[np.argmin(all_combinations_mse)][:, 0]

    sorted_idx_reverse_dict = {k: v for k, v in enumerate(sorted_idx)}

    best_combination_indexes_reversed = [sorted_idx_reverse_dict[k] for k in best_combination_indexes]

    return best_combination_indexes_reversed


def get_intersections_between_clusters(cluster1, cluster2, img_shape):
    intersections = np.empty((len(cluster1), len(cluster2), 2), dtype=np.int32)
    for i, line_1 in enumerate(cluster1):
        for j, line_2 in enumerate(cluster2):
            intersections[i][j] = intersection(line_1, line_2, img_shape)
    return intersections


def get_coordinate_boxes(tab):
    coordinate_boxes = []
    for i in range(0, 8):
        for j in range(0, 8):
            # sommet 1:
            x1 = tab[i][j][0]
            y1 = tab[i][j][1]

            # sommet 2 :
            x2 = tab[i+1][j][0]
            y2 = tab[i+1][j][1]

            # sommet 3:
            x3 = tab[i+1][j+1][0]
            y3 = tab[i+1][j+1][1]

            # sommet 4 :
            x4 = tab[i][j+1][0]
            y4 = tab[i][j+1][1]

            coordinate_boxes.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    return coordinate_boxes


