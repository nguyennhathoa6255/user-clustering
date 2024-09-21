# cleaning
import nltk
from nltk.corpus import stopwords
import re
import string
import emoji
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# image to text
import os
import requests

# Loại mấy từ không có nghĩa
time_adverbs = [
    "now", "time","year","month","week","weekend","day","hour","minute","second","today", "tonight", "yesterday", "tomorrow", "last night",
    "this morning", "this afternoon", "this evening", "next week",
    "last week", "next month", "last month", "currently", "recently",
    "lately", "soon", "early", "late", "before", "after", "soon", "already",
    "recently", "recently", "just", "yet", "still", "so far", "up to now",
    "since", "recently", "ever", "never", "always", "sometimes", "often",
    "frequently", "occasionally", "rarely", "seldom", "usually", "generally",
    "regularly", "daily", "weekly", "monthly", "yearly", "annually",  "absolutely", "actually", "almost", "always", "apparently", "approximately",
    "badly", "basically", "carefully", "certainly", "clearly", "completely",
    "constantly", "definitely", "easily", "effectively", "entirely", "especially",
    "eventually", "exactly", "extremely", "fairly", "finally", "frequently",
    "generally", "gently", "hardly", "hopefully", "immediately", "initially",
    "nearly", "necessarily", "normally", "obviously", "occasionally", "often",
    "particularly", "perfectly", "personally", "possibly", "practically",
    "presumably", "previously", "probably", "promptly", "quite", "rapidly",
    "really", "recently", "relatively", "safely", "seemingly", "seriously",
    "significantly", "simply", "slightly", "slowly", "suddenly", "surely",
    "technically", "thoroughly", "totally", "truly", "typically", "ultimately",
    "usually", "utterly", "virtually", "well", "widely"
]
quantifiers =[ "some", "many", "much", "any", "few", "several", "a few", "a lot",
    "a great deal", "plenty", "most", "more", "less", "enough", "too much",
    "too little", "all", "none", "each", "every", "either", "neither",
    "both", "half", "whole", "part", "somebody", "someone", "something",
    "nobody", "no one", "nothing", "anybody", "anyone", "anything",
    "everybody", "everyone", "everything", "few people", "a few people",
    "many people", "several people", "most people", "some people",
    "no people", "everybody", "everyone", "everything"]
def post_cleaning_Nomeaningword(text):
    text = str(text)
    list_Nonemeaningword = [word for word in text.split(' ') if word not in time_adverbs]
    cleaned_text = ' '.join(list_Nonemeaningword)

    list_Nonemeaningword = [word for word in cleaned_text.split(' ') if word not in quantifiers]
    cleaned_text = ' '.join(list_Nonemeaningword)
    return cleaned_text.strip()


#--------------
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
# Tải danh sách stop word tiếng Anh
nltk.download('stopwords')
# Lấy danh sách stop word tiếng Anh
stop_words = set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer('english', 'spain')

def post_cleaning(text):
    text = str(text)
    # Loại bỏ đường link
    link_pattern = re.compile(r"http[s]?://\S+|www\.\S+")
    cleaned_text = re.sub(link_pattern, "", text)
    # Chỉ giữ lại các danh từ, và tên riêng
    words = word_tokenize(cleaned_text)
    tagged_words = pos_tag(words)
    desired_tags = ['NN', 'NNS','NNP']
    nouns = [word for word, tag in tagged_words if tag in desired_tags]
    cleaned_text = ' '.join(nouns)
    list_Nonemeaningword = [word for word in cleaned_text.split(' ') if word not in [time_adverbs,quantifiers]]
    cleaned_text = ' '.join(list_Nonemeaningword)
    # Chuyển đổi văn bản thành chữ thường
    cleaned_text = str(cleaned_text).lower()
    # Loại bỏ các ký tự trong dấu ngoặc vuông
    cleaned_text = re.sub('\[.*?\]', '', cleaned_text)
    # Loại bỏ các thẻ HTML
    cleaned_text = re.sub('<.*?>+', '', cleaned_text)
    # Loại bỏ các dấu câu
    cleaned_text = re.sub('[%s]' % re.escape(string.punctuation), '', cleaned_text)
    # Loại bỏ các từ có chứa số
    cleaned_text = re.sub('\w*\d\w*', '', cleaned_text)
    # Loại bỏ emoji
    cleaned_text = emoji.replace_emoji(cleaned_text)
    # Loại bỏ stop word
    list_Nonestopword = [word for word in cleaned_text.split(' ') if word not in stop_words]
    cleaned_text = ' '.join(list_Nonestopword)
    # Stemming từ
    list_stemmingword = [stemmer.stem(word) for word in cleaned_text.split(' ')]
    cleaned_text = ' '.join(list_stemmingword)
    return cleaned_text.strip()

# image to text
API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
api_token = os.getenv("api_token")
headers = {"Authorization": f"Bearer {api_token}"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

# Hàm chính để xử lý tệp ảnh
def process_image(uploaded_file, query):
    if uploaded_file is not None:
        # Lấy tên tệp gốc và mở rộng
        file_name = uploaded_file.name
        file_path = os.path.join("Image", file_name)

        # Tạo thư mục nếu chưa tồn tại
        if not os.path.exists("Image"):
            os.makedirs("Image")

        # Lưu tệp vào đường dẫn
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Thực hiện truy vấn và trả về kết quả nếu file tồn tại
        try:
            output = query(file_path)
            if output:
                return output[0].get('generated_text', 'No generated text found')
            else:
                return "No output received from query."
        except Exception as e:
            return f"An error occurred: {e}"
    else:
        return "No file uploaded yet."