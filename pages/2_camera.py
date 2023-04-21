import streamlit as st
import cv2
import time
from PIL import Image
import numpy as np
import chess.svg
import model_yolo2
import ChessBoard
import fonction_annexe
from streamlit_webrtc import webrtc_streamer
# Config
st.set_page_config(page_title="Reconnaissance : caméra", page_icon=':bar_chart:')

# Application title  ===================================================================================================
st.title("Reconnaissance de parties d'échec en temps réel")
# ======================================================================================================================

# ====
webrtc_streamer(key="sample")
# ====

# Initialisez une variable pour stocker l'état de la case à cocher
checkbox_state = st.checkbox("Activer la caméra")

# Define a boolean flag to keep track of the model loading status
model_loaded = False
if not model_loaded:
    model = model_yolo2.load_yolo_model('best4.pt',version=5)
    model_loaded = True



# Créez un bouton qui active/désactive la case à cocher
if checkbox_state:
    FRAME_WINDOW = st.empty()
    camera = cv2.VideoCapture(0)
    camera.set(3, 1280)  # Résolution de l'affichage :
    camera.set(4, 720)

    while checkbox_state:
        _, image = camera.read()
        # frame = model_yolo.run_yolo(frame)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Partie liée à la détection :
        if image is not None:
            # Load the YOLO model only if it has not been loaded before
            boxes = model_yolo2.get_yolo_prediction(image, model)
            placements = model_yolo2.get_boxes_position(boxes)
            all_corners_in_chessboard, t = ChessBoard.get_chess_board(image)
            fen = fonction_annexe.get_placement(t, placements)

            # chessboard
            board = chess.Board(fen)
            img_chess = chess.svg.board(
                board,
                size=500)

            FRAME_WINDOW.write(f'<div style="display: flex; justify-content: center; padding-bottom: 50px ;">{img_chess}</div>',
                     unsafe_allow_html=True)




        time.sleep(5) # Pause de 10 secondes avant de lire la prochaine image
        # Vérifier si la case à cocher a été décochée
        if not checkbox_state:
            # Sortir de la boucle et arrêter la capture vidéo
            camera.release()
            break
            # Sortir de la boucle et arrêter la capture vidéo
