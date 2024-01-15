import streamlit as st
from io import TextIOWrapper, BytesIO
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer1 = AutoTokenizer.from_pretrained("toanduc/vit5-base-vietnews-summarization-finetuned")
model1 = AutoModelForSeq2SeqLM.from_pretrained("toanduc/vit5-base-vietnews-summarization-finetuned")

tokenizer2 = AutoTokenizer.from_pretrained("toanduc/bartpho-syllable-finetuned-sport-vietnamese")
model2 = AutoModelForSeq2SeqLM.from_pretrained("toanduc/bartpho-syllable-finetuned-sport-vietnamese")

tokenizer3 = AutoTokenizer.from_pretrained("toanduc/mT5_multilingual_XLSum-finetuned-sport-vietnamese-1")
model3 = AutoModelForSeq2SeqLM.from_pretrained("toanduc/mT5_multilingual_XLSum-finetuned-sport-vietnamese-1")

def main():
    
    st.title("Tóm Tắt Văn Bản L903-1")
    
    # Chọn phương thức nhập liệu từ người dùng
    input_method = st.radio("Chọn Phương Thức Nhập Liệu:", ("Tải Lên File", "Nhập Tay"))

    if input_method == "Tải Lên File":
        # Widget để tải lên file
        uploaded_file = st.file_uploader("Chọn một file", type=["txt", "csv", "pdf"])

        if uploaded_file is not None:
            content = uploaded_file.read()
            st.write(content)

        if uploaded_file is not None:
            # Đọc nội dung từ file
            text_input = io.TextIOWrapper(uploaded_file).read()
        else:
            st.warning("Vui lòng chọn một file để tải lên.")
            return
    else:
        # Widget để nhập văn bản
        text_input = st.text_area("Nhập văn bản cần tóm tắt:", "")

    if st.button("Tóm Tắt bằng ViT5"):
        # Gọi hàm tóm tắt văn bản ở đây (thay bằng code thực hiện tóm tắt)
        summarized_text = perform_summarization_t5(text_input)

        st.subheader("Kết Quả Tóm Tắt:")
        st.write(summarized_text)
    
    if st.button("Tóm Tắt bằng Bart"):
        # Gọi hàm tóm tắt văn bản ở đây (thay bằng code thực hiện tóm tắt)
        summarized_text = perform_summarization_Bart(text_input)

        st.subheader("Kết Quả Tóm Tắt:")
        st.write(summarized_text)
        
    if st.button("Tóm Tắt bằng mT5"):
        # Gọi hàm tóm tắt văn bản ở đây (thay bằng code thực hiện tóm tắt)
        summarized_text = perform_summarization_mT5(text_input)

        st.subheader("Kết Quả Tóm Tắt:")
        st.write(summarized_text)

def perform_summarization_t5(sentence):
    inputs = tokenizer1.encode("summarize: " + sentence, return_tensors="pt", max_length=512, truncation=True)

    summary_ids1 = model1.generate(inputs, max_length=200, length_penalty=2.0, num_beams=4, early_stopping=True, no_repeat_ngram_size=3)
    summary_text_t5 = tokenizer1.decode(summary_ids1[0], skip_special_tokens=True)
    
    summarized_text = summary_text_t5

    return summarized_text

def perform_summarization_Bart(sentence):
    inputs = tokenizer2.encode("summarize: " + sentence, return_tensors="pt", max_length=512, truncation=True)

    summary_ids2 = model2.generate(inputs, max_length=200, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary_text_bart = tokenizer2.decode(summary_ids2[0], skip_special_tokens=True)
    
    summarized_text = summary_text_bart
    
def perform_summarization_mT5(sentence):
    inputs = tokenizer3.encode("summarize: " + sentence, return_tensors="pt", max_length=512, truncation=True)

    summary_ids4 = model3.generate(inputs, max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary_text_mt5 = tokenizer3.decode(summary_ids4[0], skip_special_tokens=True)
    
    summarized_text = summary_text_mt5

    return summarized_text
if __name__ == "__main__":
    main()