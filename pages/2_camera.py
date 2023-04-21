import streamlit as st
import cv2
import time
from PIL import Image
import numpy as np
import chess.svg
import model_yolo2
import ChessBoard
import fonction_annexe
"""
import av
import threading
from typing import Union

from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Config
st.set_page_config(page_title="Reconnaissance : caméra", page_icon=':bar_chart:')

# Application title  ===================================================================================================
st.title("Reconnaissance de parties d'échec en temps réel")
# ======================================================================================================================

# Define a boolean flag to keep track of the model loading status
model_loaded = False
if not model_loaded:
    model = model_yolo2.load_yolo_model('best4.pt',version=5)
    model_loaded = True

FRAME_WINDOW = st.empty()

def main():
    class VideoTransformer(VideoTransformerBase):
        frame_lock: threading.Lock  # `transform()` is running in another thread, then a lock object is used here for thread-safety.
        in_image: Union[np.ndarray, None]
        out_image: Union[np.ndarray, None]

        def __init__(self) -> None:
            self.frame_lock = threading.Lock()
            self.in_image = None
            self.out_image = None

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            in_image = frame.to_ndarray(format="bgr24")

            out_image = in_image[:, ::-1, :]  # Simple flipping for example.

            with self.frame_lock:
                self.in_image = in_image
                self.out_image = out_image

            return out_image

    ctx = webrtc_streamer(key="snapshot", video_transformer_factory=VideoTransformer)

    if ctx.video_transformer:
        if st.button("Snapshot"):
            with ctx.video_transformer.frame_lock:
                in_image = ctx.video_transformer.in_image
                out_image = ctx.video_transformer.out_image

            if in_image is not None and out_image is not None:
                st.write("Output image:")
                st.image(out_image, channels="BGR")
            # Partie liée à la détection :
                # Load the YOLO model only if it has not been loaded before
                boxes = model_yolo2.get_yolo_prediction(out_image, model)
                placements = model_yolo2.get_boxes_position(boxes)
                all_corners_in_chessboard, t = ChessBoard.get_chess_board(out_image)
                fen = fonction_annexe.get_placement(t, placements)

                # chessboard
                board = chess.Board(fen)
                img_chess = chess.svg.board(
                    board,
                    size=500)

                FRAME_WINDOW.write(
                    f'<div style="display: flex; justify-content: center; padding-bottom: 50px ;">{img_chess}</div>',
                    unsafe_allow_html=True)
            else:
                st.warning("No frames available yet.")


if __name__ == "__main__":
    main() """

