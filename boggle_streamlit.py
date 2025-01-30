# boggle_streamlit.py
import streamlit as st
from boggle_solver import load_dictionary, build_prefixes, parse_board_input, find_words
import pytesseract
import cv2
from PIL import Image
import numpy as np
import sys


# Optional: Specify tesseract path if necessary
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load dictionary once with caching to improve performance
@st.cache(allow_output_mutation=True)
def get_dictionary():
    valid_words = load_dictionary()
    valid_prefixes = build_prefixes(valid_words)
    return valid_words, valid_prefixes


valid_words, valid_prefixes = get_dictionary()

st.title("🧩 Boggle Solver")

st.write("""
Enter the 4x4 Boggle board letters manually or upload an image of your Boggle board. The app will extract the letters and find all valid words.
""")

# Tabs for manual input and image upload
tab1, tab2 = st.tabs(["Manual Input", "Upload Image"])

with tab1:
    st.subheader("Manual Input")
    board_input = st.text_input(
        "Enter the board letters separated by commas:",
        value="N, A, I, V, O, I, C, I, I, Z, L, O, T, A, K, S"
    )
    if st.button("Find Words (Manual)"):
        try:
            board = parse_board_input(board_input)
            words_found = find_words(board, valid_words, valid_prefixes)
            if words_found:
                st.success(f"**Total Words Found:** {len(words_found)}")
                # Display words in a sorted table
                sorted_words = sorted(words_found, key=lambda x: (len(x), x))
                st.write("### Words Found (Sorted by Length):")
                st.write(sorted_words)
            else:
                st.warning("No words found on the board.")
        except ValueError as ve:
            st.error(f"Error: {ve}")

with tab2:
    st.subheader("Upload Image of Boggle Board")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            # Read image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Boggle Board Image.', use_column_width=True)

            # Convert PIL image to OpenCV format
            image_np = np.array(image)
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Preprocess the image for better OCR accuracy
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            # Apply adaptive thresholding to handle varying lighting conditions
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)

            # Optional: Dilate to enhance letter contours
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)

            # Find contours corresponding to letters
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            letter_boxes = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                # Filter out too small or too large contours
                if 20 < w < 100 and 20 < h < 100:
                    letter_boxes.append((x, y, w, h))

            if len(letter_boxes) != 16:
                st.warning(
                    f"Detected {len(letter_boxes)} letters. Please ensure the image is clear and the board is visible.")
            else:
                # Sort the letter boxes: top to bottom, then left to right
                letter_boxes = sorted(letter_boxes, key=lambda b: (b[1] // 10, b[0] // 10))

                extracted_letters = []
                for idx, (x, y, w, h) in enumerate(letter_boxes):
                    letter_img = thresh[y:y + h, x:x + w]
                    # Resize to improve OCR accuracy
                    letter_img = cv2.resize(letter_img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                    # Invert colors back for OCR
                    letter_img = cv2.bitwise_not(letter_img)
                    # Use pytesseract to do OCR on the letter
                    config = '--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                    letter = pytesseract.image_to_string(letter_img, config=config)
                    letter = letter.strip().upper()
                    if len(letter) != 1 or not letter.isalpha():
                        letter = '?'
                    extracted_letters.append(letter)

                # Display the extracted letters
                st.write("### Extracted Letters:")
                grid_letters = [extracted_letters[i * 4:(i + 1) * 4] for i in range(4)]
                grid_display = ""
                for row in grid_letters:
                    grid_display += " | ".join(row) + "\n"
                st.text(grid_display)

                # Check for any unidentified letters
                if '?' in extracted_letters:
                    st.error(
                        "Some letters could not be recognized. Please ensure the image is clear and letters are distinct.")
                    # Allow manual correction
                    st.write("### Correct Unrecognized Letters:")
                    corrected_letters = []
                    for idx, letter in enumerate(extracted_letters):
                        if letter == '?':
                            corrected = st.text_input(f"Letter {idx + 1}:", value="")
                            corrected_letters.append(corrected.upper())
                        else:
                            corrected_letters.append(letter)

                    if st.button("Find Words (Corrected Image)"):
                        try:
                            board_input_corrected = ', '.join(corrected_letters)
                            board = parse_board_input(board_input_corrected)
                            words_found = find_words(board, valid_words, valid_prefixes)
                            if words_found:
                                st.success(f"**Total Words Found:** {len(words_found)}")
                                # Display words in a sorted table
                                sorted_words = sorted(words_found, key=lambda x: (len(x), x))
                                st.write("### Words Found (Sorted by Length):")
                                st.write(sorted_words)
                            else:
                                st.warning("No words found on the board.")
                        except ValueError as ve:
                            st.error(f"Error: {ve}")
                else:
                    # All letters recognized correctly
                    board_input_image = ', '.join(extracted_letters)
                    if st.button("Find Words (Image)"):
                        try:
                            board = parse_board_input(board_input_image)
                            words_found = find_words(board, valid_words, valid_prefixes)
                            if words_found:
                                st.success(f"**Total Words Found:** {len(words_found)}")
                                # Display words in a sorted table
                                sorted_words = sorted(words_found, key=lambda x: (len(x), x))
                                st.write("### Words Found (Sorted by Length):")
                                st.write(sorted_words)
                            else:
                                st.warning("No words found on the board.")
                        except ValueError as ve:
                            st.error(f"Error: {ve}")

        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")