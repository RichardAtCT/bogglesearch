# boggle_streamlit.py
import streamlit as st
from boggle_solver import load_dictionary, build_prefixes, parse_board_input, find_words

# Load dictionary once
@st.cache(allow_output_mutation=True)
def get_dictionary():
    valid_words = load_dictionary()
    valid_prefixes = build_prefixes(valid_words)
    return valid_words, valid_prefixes

valid_words, valid_prefixes = get_dictionary()

st.title("🧩 Boggle Solver")

st.write("""
Enter the 4x4 Boggle board letters separated by commas (e.g., `N, A, I, V, O, I, C, I, I, Z, L, O, T, A, K, S`), and find all valid words.
""")

board_input = st.text_input("Enter the board letters:", value="N, A, I, V, O, I, C, I, I, Z, L, O, T, A, K, S")

if st.button("Find Words"):
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