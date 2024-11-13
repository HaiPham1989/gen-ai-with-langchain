from groq import Groq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from langchain.schema import Document  # Import Document class
import streamlit as st
import requests
import validators

## Streamlit App Setup
st.set_page_config(page_title="LangChain: Summarize Text from Audio File")
st.title("LangChain: Summarize Text from Audio File")

st.subheader('Summarize Audio URL')

## Get the Groq API key from user input
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

audio_url = st.text_input("Audio URL", label_visibility="collapsed")

## Set up the model using Groq API
client = Groq(api_key = groq_api_key)
summarizer_model = ChatGroq(api_key=groq_api_key, model="Gemma-7b-It")

def fetch_audio(audio_url):
    """Fetch audio data from a given URL."""
    response = requests.get(audio_url)
    if response.status_code == 200:
        with open("temp_audio.mp3", "wb") as f:
            f.write(response.content)
        return "temp_audio.mp3"
    else:
        raise Exception("Failed to download audio file.")
    
def audio_to_text(filepath):
    with open(filepath, "rb") as file:
        translation = client.audio.transcriptions.create(
            file=(filepath, file.read()),
            model="whisper-large-v3",
        )
    return translation.text

def summarize_text(text):
    """Summarize text using LangChain with GROQ's summarization model."""
    # Define the summarization prompt
    summarize_prompt_template = """
    Tóm tắt cuộc gọi điện thoại giữa nhân viên và khách hàng. 
    Tóm tắt nên bao gồm các chủ đề chính đã được thảo luận, chẳng hạn như yêu cầu hoặc vấn đề của khách hàng, các giải pháp hoặc sự hỗ trợ mà nhân viên đã cung cấp, 
    các hành động hoặc bước tiếp theo, và tổng thể thái độ của cuộc trò chuyện.
    Nội dung: {text}
    """
    evaluate_prompt = """
    Đánh giá chất lượng cuộc gọi:
    Điểm từ 0 đến 10
    A. Yêu cầu một cuộc gọi tốt
    1. Cấu trúc cuộc gọi
    - Mở đầu: Nếu khách hàng chưa nhớ tên bạn: Chào hỏi, giới thiệu tên mục đích làm sao để khách hàng nhớ được tên sales hoặc tên website batdongsan.com.vn
    - Nội dung chính: tư vấn dịch vụ, chốt nạp tiền, sử dụng dịch vụ, chăm sóc khách hàng
    - Kết thúc: Tóm tắt lại các nội dung đã trao đổi hoặc đặt lịch hẹn cho lần kế tiếp hoặc thúc giục khách hàng hành động, chào tạm biệt và chúc khách hàng thành công
    2. Kỹ năng giao tiếp + giọng nói
    - chủ động, năng lượng, lắng nghe tích cực, ngắn gọn, xúc tích, đồng cảm
    Giọng nói: 
    - truyền cảm, biết nhấn nhá, lên xuống tông giọng, điều chỉnh âm lượng hợp lý 
    B. Dấu hiệu đánh giá cuộc gọi chưa tốt
    1. Cấu trúc cuộc gọi
    - Cuộc gọi đầu tiên với khách hàng mà không nhắn lại tên mình hoặc tên website batdongsan.com.vn ở đầu hoặc cuối cuộc gọi  
    - Kết thúc cuộc gọi trước khách hàng, ko có nội dung tổng hợp hoặc ko có lời hẹn hay lời nói gì cho lần kết nối sau
    2. Kỹ năng giao tiếp + giọng nói
    - Giao tiếp thiếu chủ động, năng lượng kém, nói ko rõ ràng, không có trọng tâm, để khách hàng hiểu lầm, hiểu sai 
    - Thiếu lắng nghe, đồng cảm, nói chen, tỏ thái độ "không hài lòng"
    - Nói trống không hoặc sử dụng ngôn từ ko phù hợp, giải quyết vấn đề sai
    
    Cộng thêm tôí đa 1 điểm dành cho 1 số tình huống như:
    1. Xử lý tình huống khó
    2. Nói chuyện hay, hấp dẫn
    3. Sáng tạo
    4. Nhận được lời khen ngợi của KH
    5. Khách hàng giới thiệu khách hàng mới cho sale
    
    Mức 1 (Trừ 1- 2đ)
    Cố tình không nghe khách hàng nói/không hồi đáp
    Cố tình để khoảng trống
    Giao tiếp không có chủ/vị ngữ
    Cố tình dập máy trước
    Nói chen lời, nhấn giọng, gằn giọng/chưa kiên trì để lắng nghe hết nội dung khách hàng trình bày

    Mức 2 (Trừ 3-5đ)
    Sử dụng ngôn từ không phù hợp
    Cao giọng, cáu gắt, lớn tiếng với khách hàng
    Tiết lộ các bí mật, quy trình công ty gây ảnh hưởng
    Nói xấu đồng nghiệp, kích động khách hàng khiếu nại
    
    Dựa vào các tiêu chí Đánh giá chất lượng cuộc gọi. 
    Hãy đánh giá cuộc gọi với nội dung sau đây đưa ra nhận xét và đánh giá điểm (0-10):
    {text}
    """

    prompt = PromptTemplate(
        template=evaluate_prompt,
        input_variables=["text"]
    )
    
    # Wrap text in a Document object
    document = Document(page_content=text)

    # Set up the summarization chain
    chain = load_summarize_chain(summarizer_model, chain_type="stuff", prompt=prompt)
    summary = chain.run([document])

    return summary

if st.button("Summarize Audio Content"):
    """Summarize text using LangChain with GROQ's summarization model."""
    ## Validate inputs
    if not groq_api_key.strip() or not audio_url.strip():
        st.error("Please provide all required information.")
    elif not validators.url(audio_url):
        st.error("Please enter a valid audio file URL.")
    else:
        try:
            with st.spinner("Processing audio..."):
                # Fetch audio data from URL
                audio_data = fetch_audio(audio_url)
                
                if audio_data:
                    # Transcribe audio to text using GROQ's Whisper model
                    translation_text = audio_to_text(audio_data)
                    
                    # Summarize the transcription
                    summary = summarize_text(translation_text)
                    
                    st.success("Summary:")
                    st.write(summary)
                    st.write("--------------------------------")
                    st.write(translation_text)
        except Exception as e:
            st.exception(f"Exception: {e}")
