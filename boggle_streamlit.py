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
            # Apply Gaussian Blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # Apply adaptive thresholding to handle varying lighting conditions
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)

            # Optional: Dilate to enhance letter contours
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)

            # Detect grid lines using Hough Transform (optional)
            # Uncomment if grid detection is implemented
            # edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
            # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
            # if lines is not None:
            #     for line in lines:
            #         x1, y1, x2, y2 = line[0]
            #         cv2.line(image_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Find contours corresponding to letters
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Initialize list to store detected letter bounding boxes
            detected_letter_boxes = []

            # Temporary list to hold all possible boxes before filtering
            temp_letter_boxes = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                temp_letter_boxes.append((x, y, w, h))

            # If no contours detected, notify user
            if not temp_letter_boxes:
                st.warning("No contours detected. Please ensure the board is clear and well-lit.")
            else:
                # Sort contours based on area (largest to smallest)
                temp_letter_boxes = sorted(temp_letter_boxes, key=lambda b: b[2] * b[3], reverse=True)
                # Select top 16 contours assuming they are letters
                detected_letter_boxes = temp_letter_boxes[:16]

                # Alternatively, use grid-based extraction as below

                # Divide the image into a 4x4 grid
                height, width = thresh.shape
                cell_width = width // 4
                cell_height = height // 4

                extracted_letters = []
                for row in range(4):
                    for col in range(4):
                        x_start = col * cell_width
                        y_start = row * cell_height
                        x_end = (col + 1) * cell_width
                        y_end = (row + 1) * cell_height
                        cell_img = thresh[y_start:y_end, x_start:x_end]
                        # Resize to improve OCR accuracy
                        cell_img = cv2.resize(cell_img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                        # Invert colors back for OCR
                        cell_img = cv2.bitwise_not(cell_img)
                        # Use pytesseract to do OCR on the cell
                        config = '--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                        data = pytesseract.image_to_data(cell_img, config=config, output_type=pytesseract.Output.DICT)
                        letter = '?'
                        conf = 0
                        for i, word in enumerate(data['text']):
                            if word.strip().isalpha() and len(word.strip()) == 1:
                                letter = word.strip().upper()
                                conf = int(data['conf'][i])
                                break
                        # Set a confidence threshold (e.g., 60)
                        if conf < 60:
                            letter = '?'
                        extracted_letters.append(letter)

                        # Draw bounding box and label for visualization
                        cv2.rectangle(image_cv, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                        cv2.putText(image_cv, letter, (x_start + 5, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 2)

                # Convert back to PIL for display
                image_with_boxes = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                pil_image_with_boxes = Image.fromarray(image_with_boxes)
                st.image(pil_image_with_boxes, caption='Detected Letters with Grid Bounding Boxes',
                         use_column_width=True)

                # Display the extracted letters
                st.write("### Extracted Letters:")
                grid_letters = [extracted_letters[i * 4:(i + 1) * 4] for i in range(4)]
                grid_display = ""
                for row in grid_letters:
                    grid_display += " | ".join(row) + "\n"
                st.text(grid_display)

                # Check for any unidentified letters
                if '?' in extracted_letters:
                    st.error("Some letters could not be recognized. Please correct them below.")
                    # Allow users to correct unrecognized letters
                    corrected_letters = []
                    for idx, letter in enumerate(extracted_letters):
                        if letter == '?':
                            corrected = st.text_input(f"Letter {idx + 1} (Row {idx // 4 + 1}, Column {idx % 4 + 1}):",
                                                      value="")
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