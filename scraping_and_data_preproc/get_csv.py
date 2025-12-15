"""
Forming reviews.csv from ./data/ folder
"""

import pandas as pd
import os
from bs4 import BeautifulSoup

csv_dir = './csv/'
data_dir = './data/'

if(not os.path.exists(csv_dir)):
    os.makedirs(csv_dir)
    

reviews = {}

def process_reviews(content):
    soup = BeautifulSoup(content, 'html.parser')
    soup = soup.select('div.visible-xs > .media-text')
    
    data = []
    
    for sp in soup:
        if('span' in str(sp)):
            soup_span = BeautifulSoup(str(sp), 'html.parser')
            
            if soup_span.find('span', class_='full_review hidden') is not None:
                data.append(soup_span.find('span', class_='full_review hidden').text.strip())
        else:
            if sp.text is not None:
                data.append(sp.text.strip())
    
    return data

all_reviews = []
print("Starting..")

for restaurant_id in os.listdir(data_dir):
    restaurant_path = os.path.join(data_dir, restaurant_id)
    
    print(f"Restaurant: {restaurant_id}")
    
    for file in os.listdir(restaurant_path):
        if not file.endswith('.html'):
            continue
        
        full_path = os.path.join(restaurant_path, file)
        
        try:
        
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            reviews_list = process_reviews(content)
        
        except:
            print("occured an error: ")
            print(f"Path: {full_path}")
            
            # soup = BeautifulSoup(content, 'html.parser')
            # soup = soup.select('div.visible-xs > .media-text')
            
            # data = []
            
            # for sp in soup:
            #     print(sp)
            #     print("8"*44)
            #     if('span' in str(sp)):
            #         soup_span = BeautifulSoup(str(sp), 'html.parser')
            #         data.append(soup_span.find('span', class_='full_review hidden').text)
            #     else:
            #         data.append(sp.text.strip())
            
            
            # for i in data:
            #     print(i)
            #     print()
            
        
        parts = file.split('_')
        rating = int(parts[1])
        
        for review_text in reviews_list:
            all_reviews.append({
                'restaurant_id': restaurant_id,
                'rating': rating,
                'review': review_text
            })


df = pd.DataFrame(all_reviews)
df.to_csv(os.path.join(csv_dir, 'reviews.csv'), index=0)

# for root, dirs, files in os.walk(data_dir):
#     for dir in dirs:
#         for file in files:
            
#             full_path = os.path.join(root, file)
            
#             with open(full_path, 'r', encoding='utf-8') as f:
#                 content = f.read()

#             reviews_list = process_reviews(content)
            
#             # приклад: 'rest1_5.html' → name='rest1', rating='5'
#             # name_parts = file.replace('.html', '').split('_')
#             # name = name_parts[0]
#             filename = file.split('_')
#             rating = rating[1]

#             for r in reviews_list:
#                 all_reviews.append({
#                     'restaurant': dir,
#                     'rating': rating,
#                     'review': r
#                 })
        
                








# with open('elem.html', 'r', encoding='utf-8') as f:
#     content = f.read()

# soup = BeautifulSoup(content, 'html.parser')
# soup = soup.select('div.visible-xs > .media-text')
# # ('div', class_='visible-x > smedia-text')
# print(len(soup))

# data = []




# for sp in soup:
#     if('span' in str(sp)):
#         soup_span = BeautifulSoup(str(sp), 'html.parser')
#         data.append(soup_span.find('span', class_='full_review hidden').text)
#     else:
#         data.append(sp.text)


# # data = soup.find_all('div', class_='media-text')
# # print(data)
# for d in data:
#     print(d.strip())
#     print()
# print(len(data))
# # pd.DataFrame(data).to_csv(os.path.join(data_dir, 'reviews.csv'), index=False, encoding='utf-8')
