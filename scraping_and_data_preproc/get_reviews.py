"""
Getting reviews from site using requests in ./data/ folder
"""

import pandas as pd
import os
import requests
import json
from bs4 import BeautifulSoup

# creating folder for data
data_dir = './data/'
if(not os.path.exists('./data/')):
    os.makedirs('./data/')


def get_reviews(url, data_id, page=1, rating_value=1):
    
    # forming url request 
    full_url = f'{url}{data_id}/reviews?ratingValue={rating_value}&page={page}'
    
    # getting html from full_url    
    response = requests.get(full_url)
    if response.status_code == 200:
        return response.text
    else:
        return None

url = 'https://top20.ua/company-widget/'

with open('./restaurant_ids.json', 'r', encoding='utf-8') as f:
    restaurant_ids = json.load(f)

ratings = [1, 2, 5]

# in one page there are approximately 7 reviews.
# I want to get aroung 100 reviews, so 15 pages is enough

N = 15
pages = [i for i in range(N)]

for id in restaurant_ids:
    
    work_dir = os.path.join(data_dir, id)
    # creating path with id of restaurant
    if(not os.path.exists(work_dir)):
        os.makedirs(work_dir)
    
    for rating in ratings:
        for page in pages:
            reviews_raw = get_reviews(url, id, page=page, rating_value=rating)
            
            if(reviews_raw == None):
                break
            page_path = os.path.join(work_dir, f"rating_{rating}_page_{page}.html")
            with open(page_path, "w", encoding="utf-8") as f:
                f.write(reviews_raw)


soup = BeautifulSoup(reviews_raw, 'html.parser')

print(soup.find_all('div', class_='media-text'))
with open("elem2.html", "w", encoding='utf-8') as f:
    f.write(str(soup))
    
    
## коли шукаємо клас .media-text потрібно переконатися, 
# що там немає <span class="full_review hidden"></span>
# Якщо воно є, то беремо текст з нього, інакше беремо текст з .media-text