"""
Script for getting a json file with restaurant ids
"""


from playwright.sync_api import sync_playwright, TimeoutError

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    
    url = "https://top20.ua/lviv/restorani-kafe-bari/restorani/"
    page.goto(url, timeout=60000)  
    
    try:
        page.wait_for_selector(".col-md-12", timeout=60000)
    except TimeoutError:
        print("Елемент не з’явився, можливо сайт змінив структуру")

    with open("restaurants_with_ids.html", "w", encoding="utf-8") as f:
        f.write(page.content())
    
    browser.close()
