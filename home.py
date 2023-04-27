import streamlit as st
from PIL import Image
import numpy as np
import chess.svg
import model_yolo2
import ChessBoard
import fonction_annexe
from streamlit_image_select import image_select

# Config
st.set_page_config(page_title="Reconnaissance : images", page_icon=':bar_chart:')

# Application title  ===================================================================================================
st.title("Application de reconnaissance de partie d'échec")
# ======================================================================================================================
type_image = ['Utiliser une image pré-téléchargée','Télécharger une image']
choix_du_telechargement = st.radio("Télécharger une image :",type_image)
if choix_du_telechargement== 'Utiliser une image pré-téléchargée':
    uploaded_file = image_select("Choisir une image pré-téléchargée",
                       ['img_test8.png', 'img_test7.png', 'img_test9.png','before.jpg','test.png'],
                       return_value="original")

else:
    uploaded_file = st.file_uploader("Télécharger une image", type=['jpg', 'jpeg', 'png'])

# Define a boolean flag to keep track of the model loading status
model_loaded = False
if not model_loaded:
    model = model_yolo2.load_yolo_model('best4.pt',version=5)
    model_loaded = True

# Print image :
if uploaded_file is not None:
    st.write("##### Affichage de l'image téléchargée :")
    image = Image.open(uploaded_file)
    image = np.array(image)
    st.image(image, caption='Image téléchargée.')

else:
    st.warning("Téléchargez une image pour commencer. Attention, chosissez des images similaires à celles proposées dans les images pré-téléchargées.")

if uploaded_file is not None:
    # Load the YOLO model only if it has not been loaded before
    boxes = model_yolo2.get_yolo_prediction(image, model)
    placements = model_yolo2.get_boxes_position(boxes)
    all_corners_in_chessboard, t = ChessBoard.get_chess_board(image)
    fen = fonction_annexe.get_placement(t, placements)

    #st.image(detection, caption='Image téléchargée.')
    if st.button("Afficher la prédiction"):
        st.write("##### Affichage de la détection :")
        image_prediction = model_yolo2.draw_yolo_prediction(image,boxes,model)
        image_prediction = ChessBoard.draw_chess_board(image_prediction, all_corners_in_chessboard)
        st.image(image_prediction, caption="Détection de l'échiquier et des pièces.")

    st.write("##### Transcription numérique :")
    #chessboard
    board = chess.Board(fen)
    img_chess = chess.svg.board(
        board,
        size=500)

    st.write(f'<div style="display: flex; justify-content: center; padding-bottom: 50px ;">{img_chess}</div>', unsafe_allow_html=True)

    st.write("##### FEN de la partie :")
    st.write(f'<div style="border: 1px solid #e4e6f1; padding: 10px; display: flex; justify-content: center;"><strong>{fen}</strong></div>', unsafe_allow_html=True)

    st.write("##### Analyse de la partie sur Lichess :")
    link = 'https://lichess.org/analysis/standard/' + fen
    st.markdown("Analyse de la partie avec Lichess [link](%s)" % link)
