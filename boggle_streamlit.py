# boggle_streamlit.py
import streamlit as st
from boggle_solver import load_dictionary, build_prefixes, parse_board_input, find_words
import pytesseract
import cv2
from PIL import Image
import numpy as np
import sys
import logging

# Optional: Specify tesseract path if necessary
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of log messages
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("boggle_solver.log"),  # Log to a file
        logging.StreamHandler()  # Also log to console (visible in terminal)
    ]
)

logger = logging.getLogger(__name__)


# Load dictionary once with caching to improve performance
@st.cache(allow_output_mutation=True)
def get_dictionary():
    logger.info("Loading dictionary...")
    valid_words = load_dictionary()
    if not valid_words:
        logger.error("Failed to load dictionary.")
    else:
        logger.info(f"Dictionary loaded with {len(valid_words)} words.")
    valid_prefixes = build_prefixes(valid_words)
    logger.info(f"Built {len(valid_prefixes)} prefixes.")
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
        logger.info("Manual input initiated.")
        try:
            board = parse_board_input(board_input)
            logger.debug(f"Parsed board: {board}")
            words_found = find_words(board, valid_words, valid_prefixes)
            logger.info(f"Words found: {len(words_found)}")
            if words_found:
                st.success(f"**Total Words Found:** {len(words_found)}")
                # Display words in a sorted table
                sorted_words = sorted(words_found, key=lambda x: (len(x), x))
                st.write("### Words Found (Sorted by Length):")
                st.write(sorted_words)
                logger.debug(f"Sorted words: {sorted_words}")
            else:
                st.warning("No words found on the board.")
                logger.info("No words found.")
        except ValueError as ve:
            st.error(f"Error: {ve}")
            logger.error(f"ValueError: {ve}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logger.exception("Unexpected error during manual input processing.")

with tab2:
    st.subheader("Upload Image of Boggle Board")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        logger.info("Image upload initiated.")
        try:
            # Read image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Boggle Board Image.', use_column_width=True)
            logger.debug("Image uploaded and displayed.")

            # Convert PIL image to OpenCV format
            image_np = np.array(image)
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            logger.debug("Converted image to OpenCV format.")

            # Preprocess the image for better OCR accuracy
            logger.debug("Starting image preprocessing.")
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            logger.debug("Image preprocessing completed.")

            # Find contours corresponding to letters
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            logger.info(f"Detected {len(contours)} contours.")

            # Initialize list to store detected letter bounding boxes
            detected_letter_boxes = []

            # Temporary list to hold all possible boxes before filtering
            temp_letter_boxes = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                temp_letter_boxes.append((x, y, w, h))
            logger.debug(f"Initial bounding boxes: {temp_letter_boxes}")

            # Filter contours based on size
            # Adjust the size thresholds as necessary
            min_size = 20
            max_size = 100
            for box in temp_letter_boxes:
                x, y, w, h = box
                if min_size < w < max_size and min_size < h < max_size:
                    detected_letter_boxes.append(box)
            logger.info(f"Filtered bounding boxes: {detected_letter_boxes}")

            # Check if exactly 16 letters are detected
            if len(detected_letter_boxes) != 16:
                st.warning(
                    f"Detected {len(detected_letter_boxes)} letters. Please ensure the image is clear and the board is visible.")
                logger.warning(f"Expected 16 letters, but detected {len(detected_letter_boxes)}.")
            else:
                # Sort the letter boxes: top to bottom, then left to right
                detected_letter_boxes = sorted(detected_letter_boxes, key=lambda b: (b[1], b[0]))
                logger.debug("Sorted bounding boxes.")

                extracted_letters = []
                for idx, (x, y, w, h) in enumerate(detected_letter_boxes):
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
                    logger.debug(f"Extracted letter {idx + 1}: {letter}")

                    # Draw bounding box and label for visualization
                    cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image_cv, letter, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Convert back to PIL for display
                image_with_boxes = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                pil_image_with_boxes = Image.fromarray(image_with_boxes)
                st.image(pil_image_with_boxes, caption='Detected Letters with Bounding Boxes', use_column_width=True)
                logger.debug("Displayed image with bounding boxes.")

                # Display the extracted letters
                st.write("### Extracted Letters:")
                grid_letters = [extracted_letters[i * 4:(i + 1) * 4] for i in range(4)]
                grid_display = ""
                for row in grid_letters:
                    grid_display += " | ".join(row) + "\n"
                st.text(grid_display)
                logger.debug(f"Extracted letters grid: {grid_letters}")

                # Check for any unidentified letters
                if '?' in extracted_letters:
                    st.error("Some letters could not be recognized. Please correct them below.")
                    logger.error("Unrecognized letters detected.")
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
                        logger.info("User initiated word search with corrected letters.")
                        try:
                            board_input_corrected = ', '.join(corrected_letters)
                            board = parse_board_input(board_input_corrected)
                            logger.debug(f"Corrected board: {board}")
                            words_found = find_words(board, valid_words, valid_prefixes)
                            logger.info(f"Words found: {len(words_found)}")
                            if words_found:
                                st.success(f"**Total Words Found:** {len(words_found)}")
                                # Display words in a sorted table
                                sorted_words = sorted(words_found, key=lambda x: (len(x), x))
                                st.write("### Words Found (Sorted by Length):")
                                st.write(sorted_words)
                                logger.debug(f"Sorted words: {sorted_words}")
                            else:
                                st.warning("No words found on the board.")
                                logger.info("No words found after correction.")
                        except ValueError as ve:
                            st.error(f"Error: {ve}")
                            logger.error(f"ValueError during corrected input processing: {ve}")
                        except Exception as e:
                            st.error(f"An unexpected error occurred: {e}")
                            logger.exception("Unexpected error during corrected input processing.")
                else:
                    # All letters recognized correctly
                    board_input_image = ', '.join(extracted_letters)
                    if st.button("Find Words (Image)"):
                        logger.info("User initiated word search with image-extracted letters.")
                        try:
                            board = parse_board_input(board_input_image)
                            logger.debug(f"Parsed board from image: {board}")
                            words_found = find_words(board, valid_words, valid_prefixes)
                            logger.info(f"Words found: {len(words_found)}")
                            if words_found:
                                st.success(f"**Total Words Found:** {len(words_found)}")
                                # Display words in a sorted table
                                sorted_words = sorted(words_found, key=lambda x: (len(x), x))
                                st.write("### Words Found (Sorted by Length):")
                                st.write(sorted_words)
                                logger.debug(f"Sorted words: {sorted_words}")
                            else:
                                st.warning("No words found on the board.")
                                logger.info("No words found from image-extracted letters.")
                        except ValueError as ve:
                            st.error(f"Error: {ve}")
                            logger.error(f"ValueError during image input processing: {ve}")
                        except Exception as e:
                            st.error(f"An unexpected error occurred: {e}")
                            logger.exception("Unexpected error during image input processing.")
        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")
            logger.exception("Error during image upload and processing.")