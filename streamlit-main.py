import streamlit as st
from keras.models import load_model
from PIL import ImageOps, Image
import numpy as np
import tensorflow as tf
import requests, random, shutil, io
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from streamlit_image_select import image_select
from platform import system as RUNNING_OS
#from nsfw_detector import predict as nsfw

def refresh_results():
    st.session_state.new_results=True

if "new_results" not in st.session_state:
    st.session_state.new_results=False
if "files" not in st.session_state:
    st.session_state.files=[]

@st.cache_resource
def get_AI_model():
    AI = load_model("ai.h5")
    return AI

#@st.cache_resource
#def get_nsfw_model():
#    NSFW  = nsfw.load_model('./nsfw_mobilenet2.224x224.h5')
#    return NSFW

@st.cache_resource
def get_res50_model():
    from zipfile import ZipFile
    z= ZipFile('res50.zip')
    z.extractall()
    res50 = load_model("res50.keras")
    return res50
  
res_map = {i:j for i,j in enumerate(("impressionism", "renaissance","surrealism","art_nouveau","baroque","expressionism","romanticism",
          "ukiyo_e","post_impressionism","realism","AI_impressionism","AI_post_impressionism","AI_art_nouveau",
          "AI_surrealism","AI_ukiyo_e","AI_romanticism","AI_baroque","AI_expressionism","AI_renaissance","AI_realism"))}

def scrape_for_files(query):
    ysq = '+'.join(query.split())
    url=f"https://www.google.com/search?q={ysq}+filetype%3Ajpg&sca_esv=9cd90a64e37ec4fc&sca_upv=1&udm=2&biw=1536&bih=730&sxsrf=ADLYWIJOxqvpGyqsBYYqmRV_pRJvpYHMwg%3A1719876553212&ei=yTuDZuDKDKWNnesP-sm6mAM&ved=0ahUKEwig04u4_4aHAxWlRmcHHfqkDjMQ4dUDCBA&uact=5&oq=ai+generated+images+filetype%3Ajpg&gs_lp=Egxnd3Mtd2l6LXNlcnAiIGFpIGdlbmVyYXRlZCBpbWFnZXMgZmlsZXR5cGU6anBnSJQGUI0EWI0EcAF4AJABAJgBhgGgAYYBqgEDMC4xuAEDyAEA-AEBmAIAoAIAmAMAiAYBkgcAoAct&sclient=gws-wiz-serp"
    driver.get(url); driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
    imgResults = driver.find_elements(By.CLASS_NAME, "YQ4gaf")
    random.shuffle(imgResults)

    files = []
    for img in imgResults:
        src = img.get_attribute('src')
        if "favicon" in src or "base64" in src: continue
        image_content = requests.get(src).content
        files.append( io.BytesIO(image_content) )
        if len(files)==3: break

    return files


# Get this b!tch boy inside a class somehow
def classify(image_in, model):
    img_size = 48
    image = ImageOps.fit(image_in, (img_size, img_size), Image.Resampling.LANCZOS)
    image = np.asarray(image)/255.0
    image = image.reshape((-1, img_size, img_size, 3))

    confidence = model.predict(image)[0][0]
    return confidence

def res50_classify(image_in):
    img_size = 128
    image = ImageOps.fit(image_in, (img_size, img_size), Image.Resampling.LANCZOS)
    image = np.asarray(image)
    image = image.reshape((-1, img_size, img_size, 3))

    res50_label = res50_model.predict(image).argmax(axis=1)[0]

    return res50_label


#@st.cache_resource
def get_driver():
    if RUNNING_OS() == "Linux":
        service = Service(executable_path=shutil.which('chromedriver'))
    elif RUNNING_OS() == "Windows":
        service = Service()
    else:
        '''
        Mac users can f### themselves
        '''
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox") 
    options.add_argument("--headless")  
    options.add_argument("--disable-gpu")  
    options.add_argument("--window-size=1920,1080")  
    options.add_argument("--disable-dev-shm-usage") 
    options.add_argument("--log-level=OFF")
    options.add_argument("--disable-logging")
    driver = webdriver.Chrome(service=service, options=options)
    return driver


'# Scraping Categorizers'
'## Upload the image you suspect: '
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])
"Don't have 'em on you right now? We've got Google"

ysq = st.text_input("Your search query: ", key="name")
if st.session_state.new_results:
    if ysq:
        st.session_state.new_results = False
        with get_driver() as driver:
            st.session_state.files = scrape_for_files(ysq)

st.columns([1,1,1])[1].button('Refresh Results', on_click=refresh_results)
files = st.session_state.files


if files and file is None:
    images = [Image.open(file).convert('RGB') for file in files]
    file = image_select("Take your pick~",images=images)
    file = files[images.index(file)]


ai_model = get_AI_model()
res50_model = get_res50_model()
#nsfw_model = get_nsfw_model()

if file is not None:
    image = Image.open(file).convert('RGB')
    image = ImageOps.exif_transpose(image)

    st.image(image)

    st.markdown("---")
    st.markdown("## AI or not?")
    confidence = classify(image, ai_model)
    st.write(f"CIFAKE-AI: As a dumbass AI model I think this image is AI with a {10000*confidence//10/10}% chance")
    
    res50_label = res50_classify(image)
    confidence = res50_label/20
    st.write(f"Resnet50: This image is AI with a {10000*confidence//10/10}% chance")
    st.markdown("---")
    st.markdown("## Art Style?")
    st.write(f"Resnet50: Moreover, the style seems to be: {res_map[res50_label]}")

    st.markdown("---")
    st.markdown("## Safe for work?")
    "The NSFW model doesn't think anything cause I've offlined it!" 
    #nsfw.classify(nsfw_model,file,image_dim=image.size)
