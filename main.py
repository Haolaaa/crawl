import asyncio
import csv
import json
import logging
import re
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
)
from sentence_transformers import SentenceTransformer
import numpy as np


logging.basicConfig(level=logging.INFO)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

browser_config = BrowserConfig(
    headless=False,
    use_managed_browser=True,
    user_data_dir="/Users/ryan/my_chrome_profile",
    browser_type="chromium",
)


def calculate_similarity(query: str, product_title: str) -> float:
    """
    Calculate cosine similarity between query and product title.
    Returns a score between 0 and 1 (higher = more similar).
    """
    # encode both texts
    query_embedding = embedding_model.encode(query, convert_to_tensor=False)
    title_embedding = embedding_model.encode(product_title, convert_to_tensor=False)

    # calculate cosine similarity
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
        title = product.get("title", "")
        if not title or title == "N/A":
            continue

        similarity_score = calculate_similarity(query, title)

        if similarity_score >= threshold:
            product["similarity_score"] = round(similarity_score, 3)
            filtered_products.append(product)

    # Sort by similarity score (highest first)
    filtered_products.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)

    return filtered_products


def extract_amazon_data(html):
    soup = BeautifulSoup(html, "html.parser")
    products = []

    for item in soup.find_all("div", {"data-component-type": "s-search-result"}):
        is_sponsor = item.find(string="Sponsorowane")
        if is_sponsor:
            continue
        try:
            title_div = item.find("div", {"data-cy": "title-recipe"})
            title_span = title_div.find("span") if title_div else None
            title = title_span.get_text(strip=True) if title_span else ""

            price_whole = item.find("span", {"class": "a-price-whole"})
            price_fraction = item.find("span", {"class": "a-price-fraction"})
            price = (
                f"{price_whole.get_text(strip=True)}{price_fraction.get_text(strip=True)}"
                if price_whole and price_fraction
                else "N/A"
            )

            link_elem = item.find("a", {"class": "a-link-normal"})
            link = (
                f"https://www.amazon.pl{link_elem['href']}"
                if link_elem and "href" in link_elem.attrs
                else "N/A"
            )

            products.append(
                {
                    "title": title,
                    "price": price,
                    "link": link,
                }
            )
        except Exception as e:
            print(f"Error parsing product: {e}")
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
                    products.append({"title": title, "price": price, "link": link})

            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
            except KeyError:
                continue

    return products


def extract_pepper_data(html):
    soup = BeautifulSoup(html, "html.parser")
    products = []

    for el in soup.find_all(attrs={"data-vue3": True}):
        data_attr = el.get("data-vue3")

        try:
            data_json = json.loads(data_attr)
            thread = data_json.get("props", {}).get("thread", {})

            title = thread.get("title")
            link = thread.get("shareableLink")
            price = thread.get("price")

            if title and link and price:
                products.append({"title": title, "link": link, "price": price})

        except Exception as e:
            print(f"Error parsing product: {e}")
            continue

    if len(products) == 0:
        for item in soup.select('article[class*="thread"]'):
            try:
                container = item.find("div", {"class": "threadListCard-body"})
                if not container:
                    continue
                link = ""
                price = ""

                name_elem = container.select_one(
                    'a[class*="js-thread-title"], span[class*="js-thread-title"]'
                )

                if not name_elem:
                    continue
                title = name_elem.get_text(strip=True)
                link = name_elem.get("href", "")

                box = container.find("div", {"class": "box--contents"})

                if box:
                    price_tag = box.select_one(
                        "div.flex--inline span.vAlign--all-tt span.color--text-NeutralSecondary"
                    )
                    if price_tag:
                        price_text = price_tag.get_text(strip=True).replace("\xa0", "")
                        price = extract_price(price_text)

                if link == "":
                    link_elem = item.select_one('a[class*="thread-link"], a[href*="/"]')
                    url = link_elem.get("href", "") if link_elem else ""
                    link = clean_url(url, "https://www.pepper.pl")

                products.append({"title": title, "link": link, "price": price})
            except Exception as e:
                print(f"Error parsing product: {e}")
                continue

        return products

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
                    products.append({"title": title, "link": link, "price": price})

            except json.JSONDecodeError as e:
                print(f"Error parsing product: {e}")
                continue
    return products


def extract_price(text: str) -> Optional[float]:
    """
    Extract numeric price from text
    Handles formats like: 1 299,99 zł, 1299.99, 1.299,99
    """
    if not text:
        return None

    # Remove currency symbols and extra text
    text = text.replace("zł", "").replace("PLN", "").replace("€", "")
    text = text.replace("\xa0", " ").strip()

    # Find number patterns
    # Matches: 1299.99, 1 299,99, 1.299,99
    patterns = [
        r"(\d+[\s\.]?\d*,\d{2})",  # 1 299,99 or 1.299,99
        r"(\d+\.\d{2})",  # 1299.99
        r"(\d+)",  # 1299
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            price_str = match.group(1)
            # Normalize: remove spaces, replace comma with dot
            price_str = price_str.replace(" ", "").replace(".", "").replace(",", ".")
            try:
                return float(price_str)
            except ValueError:
                continue

    return None


# Platform configuration
PLATFORMS = {
    "amazon": {
        "name": "Amazon",
        "url_template": "https://www.amazon.pl/s?k={query}",
        "query_formatter": lambda q: q.replace(" ", "+"),
        "extractor": extract_amazon_data,
        "enabled": True,
        "wait_for": "css:div[data-component-type='s-search-result']",
    },
    "allegro": {
        "name": "Allegro",
        "url_template": "https://allegro.pl/listing?string={query}",
        "query_formatter": lambda q: q.replace(" ", "%20"),
        "extractor": extract_allegro_data,
        "enabled": True,
        "wait_for": "css:div[class='opbox-listing']",
    },
    "pepper": {
        "name": "Pepper",
        "url_template": "https://www.pepper.pl/search?q={query}",
        "query_formatter": lambda q: q.replace(" ", "%20"),
        "extractor": extract_pepper_data,
        "enabled": True,
        "wait_for": "css:.js-threadList",
    },
    "ceneo": {
        "name": "Ceneo",
        "url_template": "https://www.ceneo.pl/;szukaj-{query}",
        "query_formatter": lambda q: q.replace(" ", "+"),
        "extractor": extract_ceneo_data,
        "enabled": True,
        "wait_for": "css:.category-list",
    },
}


async def main():
    crawl_config = CrawlerRunConfig(
        magic=True,
        simulate_user=True,
        delay_before_return_html=2.0,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        product_name = "iPhone 16 Pro 256 GB"
        SIMILARITY_THRESHOLD = 0.8
        platform_results = {}

        for platform_id, platform_config in PLATFORMS.items():
            if not platform_config["enabled"]:
                print(f"⏭️  Skipping {platform_config['name']} (disabled)")
                continue
            crawl_config.wait_for = platform_config["wait_for"]

            formatted_query = platform_config["query_formatter"](product_name)
            url = platform_config["url_template"].format(query=formatted_query)

            print(f"🔍 Searching {platform_config['name']}...")
            result = await crawler.arun(url=url, config=crawl_config)

            if not result.success:
                print(f"❌ Error on {platform_config['name']}: {result.error_message}")
                platform_results[platform_id] = []
            else:
                products = platform_config["extractor"](result.html)
                filtered_products = filter_products_by_similarity(
                    products, product_name, SIMILARITY_THRESHOLD
                )
                platform_results[platform_id] = filtered_products
                print(f"✅ Found {len(products)} products on {platform_config['name']}")
                print(
                    f"   🎯 {len(filtered_products)} products match after filtering (threshold: {SIMILARITY_THRESHOLD})"
                )

        # Find cheapest product from each platform
        def get_cheapest(products, platform_name):
            if not products:
                return {
                    "platform": platform_name,
                    "title": "N/A",
                    "link": "N/A",
                    "price": float("inf"),
                }

            # Parse price strings to floats for comparison
            def parse_price(price_str):
                if price_str == "N/A" or price_str is None:
                    return float("inf")
                try:
                    # Remove currency symbols and whitespace, replace comma with dot
                    cleaned = (
                        str(price_str)
                        .replace(",", ".")
                        .replace("zł", "")
                        .replace("PLN", "")
                        .strip()
                    )
                    return float(cleaned)
                except (ValueError, AttributeError):
                    return float("inf")

            cheapest = min(products, key=lambda p: parse_price(p.get("price", "N/A")))
            return {
                "platform": platform_name,
                "title": cheapest.get("title", "N/A"),
                "link": cheapest.get("link", "N/A"),
                "price": parse_price(cheapest.get("price", "N/A")),
            }

        # Get cheapest from each platform
        platform_cheapest = {}
        for platform_id, platform_config in PLATFORMS.items():
            if platform_config["enabled"]:
                products = platform_results.get(platform_id, [])
                platform_cheapest[platform_id] = get_cheapest(
                    products, platform_config["name"]
                )

        # Find overall cheapest
        all_cheapest = list(platform_cheapest.values())
        overall_cheapest = (
            min(all_cheapest, key=lambda p: p["price"]) if all_cheapest else None
        )
        overall_cheapest_price = (
            overall_cheapest["price"]
            if overall_cheapest and overall_cheapest["price"] != float("inf")
            else "N/A"
        )

        # Write to CSV - dynamically generate fields based on enabled platforms
        with open("price_comparison.csv", "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = []
            for platform_id in PLATFORMS.keys():
                if PLATFORMS[platform_id]["enabled"]:
                    fieldnames.extend([f"{platform_id}_link", f"{platform_id}_price"])
            fieldnames.extend(["overall_cheapest_price", "overall_cheapest_platform"])

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Build row data
            row_data = {}
            for platform_id, cheapest_data in platform_cheapest.items():
                row_data[f"{platform_id}_link"] = cheapest_data["link"]
                price_value = (
                    cheapest_data["price"]
                    if cheapest_data["price"] != float("inf")
                    else "N/A"
                )
                row_data[f"{platform_id}_price"] = price_value

            row_data["overall_cheapest_price"] = overall_cheapest_price
            row_data["overall_cheapest_platform"] = (
                overall_cheapest["platform"] if overall_cheapest else "N/A"
            )

            writer.writerow(row_data)

        print("\n✅ Price comparison saved to price_comparison.csv")
        if overall_cheapest:
            print(
                f"🏆 Overall cheapest: {overall_cheapest['platform']} at {overall_cheapest_price} PLN"
            )


if __name__ == "__main__":
    asyncio.run(main())
