# Palette's Journal

## 2024-05-22 - Initial Setup
**Learning:** This is a Streamlit application. UX improvements in Streamlit often involve using `st.help`, `st.expander` for context, `placeholder` text in inputs, and ensuring efficient data loading feedback with `st.spinner` or `st.progress`. CSS customization is possible but should be minimal (`st.markdown` with `unsafe_allow_html=True` is common but should be used sparingly).
**Action:** Focus on Streamlit-native UX enhancements like better empty states, clearer instructions using `help` parameters in widgets, and improved error handling/feedback.

## 2024-05-22 - Login Experience
**Learning:** Standard Streamlit `text_input` widgets do not submit on Enter unless wrapped in a `st.form`. This breaks the expected "Type Password -> Enter -> Login" flow.
**Action:** Always wrap single-input forms (like login or search) in `st.form` to enable keyboard submission, improving accessibility and reducing friction.
