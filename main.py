import json
import logging
from typing import Dict, List
import sys
import os

import numpy as np
from bs4 import BeautifulSoup
from playwright.async_api import Page
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


model_path = resource_path("model_data")
if os.path.exists(model_path):
    print(f"Loading model from local path: {model_path}")
    embedding_model = SentenceTransformer(model_path)
else:
    print("Local model not found, downloading...")
    embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")


def calculate_similarity(query: str, product_title: str) -> float:
    """
    Calculate cosine similarity between query and product title.
    Returns a score between 0 and 1 (higher = more similar).
    """

    query_embedding = embedding_model.encode(query, show_progress_bar=False)
    title_embedding = embedding_model.encode(product_title, show_progress_bar=False)

    similarity = np.dot(query_embedding, title_embedding) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(title_embedding)
    )

    return float(similarity)


def filter_products_by_similarity(
    products: List[Dict], query: str, threshold: float = 0.5
) -> List[Dict]:
    """
    Filter products based on semantic similarity to the query.

    Args:
        products: List of product dictionaries
        query: Original search query
        threshold: Minimum similarity score (0-1) to keep a product

    Returns:
        Filtered list of products with similarity scores added
    """
    filtered_products = []

    for product in products:
        title: str = product.get("title", "")
        if not title:
            continue

        similarity_score = calculate_similarity(
            query.lower(), title.replace("(", "").replace(")", "").lower()
        )

        if similarity_score > 0.85:
            print(f"title: {title}")
            print(f"query: {query}")
            print(f"similarity_score: {similarity_score}")

        if similarity_score >= threshold:
            product["similarity_score"] = round(similarity_score, 3)
            filtered_products.append(product)

    # Sort by similarity score (highest first)
    filtered_products.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)

    return filtered_products


def extract_amazon_data(html):
    soup = BeautifulSoup(html, "html.parser")
    products = []

    _item = {}
    for item in soup.find_all("div", {"data-component-type": "s-search-result"}):
        is_sponsor = item.find(string="Sponsorowane")
        if is_sponsor:
            continue
        try:
            _item = item

            title_div = item.find("div", {"data-cy": "title-recipe"})
            title_span = title_div.find("span") if title_div else None
            title = title_span.get_text() if title_span else ""

            price_whole = item.find("span", {"class": "a-price-whole"})
            price_fraction = item.find("span", {"class": "a-price-fraction"})
            price = (
                f"{price_whole.get_text()}{price_fraction.get_text()}"
                if price_whole and price_fraction
                else "0.0"
            )

            link_elem = item.find("a", {"class": "a-link-normal"})
            link = f"https://www.amazon.pl{link_elem['href']}"

            products.append(
                {
                    "title": title,
                    "price": parse_price(price),
                    "link": link,
                }
            )
        except Exception as e:
            print(f"Error parsing product({_item}): {e}")
            continue

    return products


def extract_allegro_data(html):
    soup = BeautifulSoup(html, "html.parser")
    products = []

    for script in soup.find_all("script"):
        if script.string and "__listing_StoreState" in script.string:
            try:
                data = json.loads(script.string)
                elements = data["__listing_StoreState"]["items"]["elements"]

                for idx, el in enumerate(elements):
                    if el.get("type") == "banner":
                        continue

                    title = el.get("alt")
                    price = el.get("price", {}).get("mainPrice", {}).get("amount")
                    link = el.get("url")
                    products.append(
                        {"title": title, "price": parse_price(price), "link": link}
                    )

            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
            except KeyError:
                continue

    return products


def extract_pepper_data(html):
    """
    Parses HTML to find product list using Vue3 data attributes
    or fallback DOM scraping.
    """
    soup = BeautifulSoup(html, "html.parser")
    products = []

    # Method 1: Try parsing the JSON in data-vue3 attributes (More reliable)
    for el in soup.find_all(attrs={"data-vue3": True}):
        data_attr = el.get("data-vue3")

        try:
            data_json = json.loads(data_attr)
            # navigate the nested structure safely
            thread = data_json.get("props", {}).get("thread", {})

            title = thread.get("title")
            link = thread.get("shareableLink")
            price_raw = thread.get("price")
            temperature = thread.get("temperature")

            if title:
                # We calculate numeric price for sorting, but keep raw for display if needed
                numeric_price = parse_price(str(price_raw)) if price_raw else 0.0

                products.append(
                    {
                        "title": title,
                        "link": link,
                        "price": numeric_price,
                        "display_price": price_raw,
                        "temperature": temperature,
                    }
                )

        except Exception as e:
            print(f"Error parsing Vue data: {e}")
            continue

    # Method 2: Fallback to DOM scraping if Vue method yielded nothing
    if len(products) == 0:
        for item in soup.select('article[class*="thread"]'):
            try:
                container = item.find("div", {"class": "threadListCard-body"})
                if not container:
                    continue

                # Title Extraction
                name_elem = container.select_one(
                    'a[class*="js-thread-title"], span[class*="js-thread-title"]'
                )
                if not name_elem:
                    continue
                title = name_elem.get_text(strip=True)

                # Link Extraction
                link_elem = item.select_one('a[class*="thread-link"], a[href*="/"]')
                raw_link = link_elem.get("href", "") if link_elem else ""
                link = clean_url(raw_link, "https://www.pepper.pl")

                # Price Extraction
                price = 0.0
                price_text = "0"
                box = container.find("div", {"class": "box--contents"})
                if box:
                    price_tag = box.select_one(
                        "span.color--text-NeutralSecondary"
                    )  # Adjusted selector based on common pepper themes
                    if price_tag:
                        price_text = price_tag.get_text(strip=True)
                        price = parse_price(price_text)

                # Temperature Extraction
                temperature = 0
                vote_el = item.select_one(".cept-vote-temp span")
                if vote_el:
                    temp_text = vote_el.get_text(strip=True).replace("°", "").strip()
                    try:
                        temperature = float(temp_text)
                    except:
                        temperature = 0

                products.append(
                    {
                        "title": title,
                        "link": link,
                        "price": price,
                        "display_price": price_text,
                        "temperature": temperature,
                    }
                )
            except Exception as e:
                print(f"Error parsing DOM product: {e}")
                continue

    return products


def extract_ceneo_data(html):
    soup = BeautifulSoup(html, "html.parser")
    products = []

    for script in soup.find_all("script"):
        if script.string and "itemListElement" in script.string:
            try:
                data = json.loads(script.string)

                elements = data["itemListElement"]

                for el in elements:
                    item = el["item"]
                    if item.get("@type") != "Product":
                        continue

                    title = item["name"]
                    price = item["offers"]["lowPrice"]
                    link = item["url"]
                    products.append(
                        {"title": title, "link": link, "price": parse_price(price)}
                    )

            except json.JSONDecodeError as e:
                print(f"Error parsing product: {e}")
                continue
    return products


def clean_url(url: str, base_url: str) -> str:
    """Make sure URL is absolute"""
    if url.startswith("http"):
        return url
    elif url.startswith("//"):
        return "https:" + url
    elif url.startswith("/"):
        return base_url.rstrip("/") + url
    else:
        return base_url.rstrip("/") + "/" + url


def parse_price(text: str) -> float:
    """
    Extract numeric price from text
    Handles formats like: 1 299,99 zł, 1299.99, 1.299,99, 4 849zł, 5 390,26zł
    """
    if not text:
        return float("inf")

    if isinstance(text, float):
        return text

    text = (
        text.replace("zł", "")
        .replace("PLN", "")
        .replace("€", "")
        .replace(" ", "")
        .replace("\xa0", "")
        .replace(",", ".")
    )

    return float(text)


# Platform configuration
PLATFORMS = {
    "amazon": {
        "name": "Amazon",
        "url_template": "https://www.amazon.pl/s?k={query}",
        "query_formatter": lambda q: q.replace(" ", "+"),
        "extractor": extract_amazon_data,
        "enabled": True,
        "wait_for": "div[data-component-type='s-search-result']",
    },
    "allegro": {
        "name": "Allegro",
        "url_template": "https://allegro.pl/listing?string={query}",
        "query_formatter": lambda q: q.replace(" ", "%20"),
        "extractor": extract_allegro_data,
        "enabled": True,
        "wait_for": "div[class='opbox-listing']",
    },
    "pepper": {
        "name": "Pepper",
        "url_template": "https://www.pepper.pl/search?q={query}",
        "query_formatter": lambda q: q.replace(" ", "%20"),
        "extractor": extract_pepper_data,
        "enabled": True,
        "wait_for": ".js-threadList",
    },
    "ceneo": {
        "name": "Ceneo",
        "url_template": "https://www.ceneo.pl/;szukaj-{query}",
        "query_formatter": lambda q: q.replace(" ", "+"),
        "extractor": extract_ceneo_data,
        "enabled": True,
        "wait_for": ".category-list",
    },
}


async def scrape_platform(
    page: Page, platform_id: str, platform_config: dict, product_name: str
) -> tuple:
    """Scrape a single platform"""
    try:
        formatted_query = platform_config["query_formatter"](product_name)
        url = platform_config["url_template"].format(query=formatted_query)

        print(f"🔍 Navigating to {platform_config['name']}...")

        await page.goto(url, wait_until="domcontentloaded", timeout=30000)

        try:
            await page.wait_for_selector(
                platform_config["wait_for"], timeout=60 * 60 * 1000
            )
        except Exception as e:
            print(f"⚠️  Timeout waiting for selector on {platform_config['name']}: {e}")

        html = await page.content()

        return (platform_id, {"success": True, "html": html, "error": None})

    except Exception as e:
        print(f"❌ Error scraping {platform_config['name']}: {str(e)}")
        return (platform_id, {"success": False, "html": None, "error": str(e)})
