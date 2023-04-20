import streamlit as st
import cv2
from shapely.geometry import Point, Polygon

# Fonction pour obtenir la hauteur de l'écran


def get_screen_height():
    return st.experimental_get_query_params().get("height", [None])[0]

# Fonction pour load une image
def load_image(filename):
    img = cv2.imread(filename)
    return img


def rotate(s):
    output = ''
    next_line = 0
    for i in range(0,8):
        k = 0
        for j in range(0,8):
            output += s[8*j + next_line + k]
            k +=1
        next_line += 1
        if i!=7:
            output += '/'
    return output



def compress_string(s):
    compressed_string = ''
    count = 0
    for char in s:
        if char == 'x':
            count += 1
        elif count != 0 :
            compressed_string += str(count) + char
            count = 0
        else:
            compressed_string += char
    if count != 0:
        compressed_string += str(count)
    return compressed_string

def get_placement(coordinate_cases, coordinate_pieces):
    fen = ""
    case_vide = 0
    i = 0
    # On parcourt chaque chase de l'echiquier
    for case in coordinate_cases:
        if i % 8 == 0 and i != 0:
            fen += '/'
            i = 0
        polygon = Polygon([case[0], case[1], case[2], case[3]])
        # Puis chaque coordonnees des pieces detectees
        is_piece = False # On initialise une variable d'état quant à la présence d'une pièce
        for piece in coordinate_pieces:
            cls = piece[2]
            point = Point(piece[0], piece[1])
            # Dans le cas ou il y a match entre une case et une piece
            if polygon.contains(point):
                fen = fen + class_to_fen(cls)
                is_piece = True # On actualise la variable d'état
                break
        if is_piece == False:
            fen += 'x'
            case_vide += 1
        i = i + 1
    fen = rotate(fen)
    fen = compress_string(fen)
    return fen



def class_to_fen(cls):
    if cls == 0:
        fen_caractere = 'K'
    elif cls == 1:
        fen_caractere = 'Q'
    elif cls == 2:
        fen_caractere = 'R'
    elif cls == 3:
        fen_caractere = 'B'
    elif cls == 4:
        fen_caractere = 'N'
    elif cls == 5:
        fen_caractere = 'P'
    elif cls == 6:
        fen_caractere = 'k'
    elif cls == 7:
        fen_caractere = 'q'
    elif cls == 8:
        fen_caractere = 'r'
    elif cls == 9:
        fen_caractere = 'b'
    elif cls == 10:
        fen_caractere = 'n'
    else:
        fen_caractere = 'p'
    return fen_caractere
