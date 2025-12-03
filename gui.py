import asyncio
import csv
import sqlite3
import sys
import json
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QTableWidget,
    QHeaderView,
    QAbstractItemView,
    QTableWidgetItem,
    QTabWidget,
)

from playwright.async_api import async_playwright

from main import (
    PLATFORMS,
    filter_products_by_similarity,
    scrape_platform,
)


class DatabaseManager:
    """Manages SQLite database operations including preferences"""

    def __init__(self, db_path: str = "price_comparison.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT NOT NULL,
                similarity_threshold REAL NOT NULL,
                search_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                overall_cheapest_platform TEXT,
                overall_cheapest_price REAL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS platform_cheapest (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                search_id INTEGER NOT NULL,
                platform TEXT NOT NULL,
                product_title TEXT,
                price REAL,
                link TEXT,
                temperature TEXT,
                FOREIGN KEY (search_id) REFERENCES searches(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def save_preference(self, key: str, value: str):
        """Save a user preference"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT OR REPLACE INTO preferences (key, value) VALUES (?, ?)",
            (key, value),
        )

        conn.commit()
        conn.close()

    def get_preference(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a user preference"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT value FROM preferences WHERE key = ?", (key,))
        result = cursor.fetchone()

        conn.close()

        return result[0] if result else default

    def save_search(
        self,
        product_name: str,
        threshold: float,
        platform_cheapest: dict,
        overall_cheapest: dict,
    ) -> int:
        """Save search results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO searches (product_name, similarity_threshold, 
                                overall_cheapest_platform, overall_cheapest_price)
            VALUES (?, ?, ?, ?)
        """,
            (
                product_name,
                threshold,
                overall_cheapest["platform"],
                overall_cheapest["price"],
            ),
        )

        search_id = cursor.lastrowid

        for platform_id, cheapest_data in platform_cheapest.items():
            cursor.execute(
                """
                INSERT INTO platform_cheapest (search_id, platform, product_title, price, link, temperature)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    search_id,
                    cheapest_data["platform"],
                    cheapest_data.get("title"),
                    cheapest_data.get("price"),
                    cheapest_data.get("link"),
                    cheapest_data.get("temperature"),
                ),
            )

        conn.commit()
        conn.close()

        return search_id

    def get_all_searches_with_platforms(self) -> List[Dict]:
        """Get all search records with cheapest from each platform"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all searches
        cursor.execute("""
            SELECT id, product_name, similarity_threshold, search_date,
                overall_cheapest_platform, overall_cheapest_price
            FROM searches
            ORDER BY search_date DESC
        """)

        searches = cursor.fetchall()

        # For each search, get platform cheapest
        result = []
        for search in searches:
            search_id = search[0]

            cursor.execute(
                """
                SELECT platform, product_title, price, link, temperature
                FROM platform_cheapest
                WHERE search_id = ?
                ORDER BY platform
            """,
                (search_id,),
            )

            platform_data = cursor.fetchall()

            search_dict = {
                "id": search[0],
                "product_name": search[1],
                "threshold": search[2],
                "date": search[3],
                "overall_platform": search[4],
                "overall_price": search[5],
                "platforms": {},
            }

            for platform, title, price, link, temperature in platform_data:
                search_dict["platforms"][platform] = {
                    "title": title,
                    "price": price,
                    "link": link,
                    "temperature": temperature,
                }

            result.append(search_dict)

        conn.close()
        return result

    def delete_search(self, search_id: int):
        """Delete a search and its results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM searches WHERE id = ?", (search_id,))
        cursor.execute(
            "DELETE FROM platform_cheapest WHERE search_id = ?", (search_id,)
        )

        conn.commit()
        conn.close()

    def export_searches_with_column_order(
        self,
        file_path: str,
        search_ids: Optional[List[int]] = None,
        visual_to_logical: Optional[List[int]] = None,
        headers_in_visual_order: Optional[List[str]] = None,
    ):
        """
        Export searches to CSV matching the exact table view format.

        Args:
            file_path: Path to save CSV
            search_ids: List of search IDs to export (None = all)
            visual_to_logical: Maps visual position to logical column index
            headers_in_visual_order: Header labels in visual order
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build WHERE clause for specific searches
        if search_ids:
            placeholders = ",".join("?" * len(search_ids))
            where_clause = f"WHERE s.id IN ({placeholders})"
            params = search_ids
        else:
            where_clause = ""
            params = []

        # Get all data
        cursor.execute(
            f"""
            SELECT s.id, s.product_name, s.similarity_threshold, s.search_date,
                s.overall_cheapest_platform, s.overall_cheapest_price,
                pc.platform, pc.product_title, pc.price, pc.link, pc.temperature
            FROM searches s
            LEFT JOIN platform_cheapest pc ON s.id = pc.search_id
            {where_clause}
            ORDER BY s.search_date DESC, s.id, pc.platform
        """,
            params,
        )

        all_data = cursor.fetchall()
        conn.close()

        if not all_data:
            return

        # Group by search_id
        searches_dict = {}
        for row in all_data:
            search_id = row[0]
            if search_id not in searches_dict:
                searches_dict[search_id] = {
                    "id": row[0],
                    "product_name": row[1],
                    "threshold": row[2],
                    "date": row[3],
                    "overall_platform": row[4],
                    "overall_price": row[5],
                    "platforms": {},
                }

            if row[6]:  # platform exists
                searches_dict[search_id]["platforms"][row[6]] = {
                    "title": row[7],
                    "price": row[8],
                    "link": row[9],
                    "temperature": row[10],
                }

        # Build rows matching table structure
        with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Write headers
            if headers_in_visual_order:
                writer.writerow(headers_in_visual_order)
            else:
                # Default header order
                base_headers = ["ID", "Product Name", "Threshold", "Date"]
                platform_headers = []
                for platform_id, platform_config in PLATFORMS.items():
                    platform_name = platform_config["name"]
                    platform_headers.extend(
                        [f"{platform_name} Link", f"{platform_name} Price", f"{platform_name} Hot"]
                    )
                final_headers = (
                    base_headers + platform_headers + ["Overall Cheapest Price"]
                )
                writer.writerow(final_headers)

            # Write data rows
            for search_id in sorted(searches_dict.keys(), reverse=True):
                search = searches_dict[search_id]

                # Build row in logical order (same as table construction)
                row_data_logical = []

                # Base columns
                row_data_logical.append(str(search["id"]))
                row_data_logical.append(search["product_name"])
                row_data_logical.append(f"{search['threshold']:.2f}")
                row_data_logical.append(search["date"])

                # Platform columns
                for platform_id, platform_config in PLATFORMS.items():
                    platform_name = platform_config["name"]
                    platform_data = search["platforms"].get(platform_name, {})

                    # Link
                    link = platform_data.get("link", "N/A")
                    row_data_logical.append(link if link != "N/A" else "N/A")

                    # Price
                    price = platform_data.get("price")
                    if price is not None:
                        row_data_logical.append(f"{price:.2f}")
                    else:
                        row_data_logical.append("N/A")

                    # temperature
                    temperature = platform_data.get("temperature")
                    if temperature is not None:
                        row_data_logical.append(temperature)
                    else:
                        row_data_logical.append("N/A")


                # Overall cheapest price
                overall_price = search["overall_price"]
                if overall_price:
                    row_data_logical.append(f"{overall_price:.2f}")
                else:
                    row_data_logical.append("N/A")

                # Reorder row according to visual order
                if visual_to_logical:
                    row_data_visual = []
                    for logical_idx in visual_to_logical:
                        if logical_idx < len(row_data_logical):
                            row_data_visual.append(row_data_logical[logical_idx])
                        else:
                            row_data_visual.append("N/A")
                    writer.writerow(row_data_visual)
                else:
                    writer.writerow(row_data_logical)


class ScraperThread(QThread):
    """Background thread for running the scraper to keep UI responsive"""

    progress_update = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        product_name: str,
        similarity_threshold: float,
        enabled_platforms: Dict[str, bool],
        user_data_dir: str,
        headless: bool = False,
    ):
        super().__init__()
        self.product_name = product_name
        self.similarity_threshold = similarity_threshold
        self.enabled_platforms = enabled_platforms
        self.user_data_dir = user_data_dir
        self.headless = headless

    def run(self):
        """Run the scraper in background thread"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(self.scrape())
            loop.close()

            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    async def scrape(self):
        """Main scraping logic using Playwright"""
        platform_results = {}

        async with async_playwright() as p:
            # Launch browser with persistent context
            browser = await p.chromium.launch_persistent_context(
                self.user_data_dir,
                headless=self.headless,
                viewport={"width": 1200, "height": 1400},
            )

            tasks = []
            platform_order = []

            for platform_id, platform_config in PLATFORMS.items():
                if not self.enabled_platforms.get(platform_id, True):
                    self.progress_update.emit(
                        f"⏭️  Skipping {platform_config['name']} (disabled)"
                    )
                    continue

                # Create a new page for each platform
                page = await browser.new_page()

                # Add some human-like behavior
                await page.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => false
                    });
                """)

                tasks.append(
                    scrape_platform(
                        page, platform_id, platform_config, self.product_name
                    )
                )
                platform_order.append(platform_id)
                self.progress_update.emit(f"🔍 Preparing {platform_config['name']}...")

            if tasks:
                self.progress_update.emit(
                    f"\n🚀 Starting concurrent search across {len(tasks)} platforms...\n"
                )
                results = await asyncio.gather(*tasks)

                for platform_id, result in results:
                    platform_config = PLATFORMS[platform_id]

                    if not result["success"]:
                        self.progress_update.emit(
                            f"❌ Error on {platform_config['name']}: {result['error']}"
                        )
                        platform_results[platform_id] = []
                    else:
                        products = platform_config["extractor"](result["html"])
                        filtered_products = filter_products_by_similarity(
                            products, self.product_name, self.similarity_threshold
                        )
                        platform_results[platform_id] = filtered_products
                        self.progress_update.emit(
                            f"✅ Found {len(products)} products on {platform_config['name']}"
                        )
                        self.progress_update.emit(
                            f"   🎯 {len(filtered_products)} products match after filtering (threshold: {self.similarity_threshold})"
                        )

            await browser.close()

        return self.process_results(platform_results)

    def process_results(self, platform_results):
        """Process and find cheapest products"""

        def get_cheapest(products, platform_name):
            if not products:
                return {
                    "platform": platform_name,
                    "title": "N/A",
                    "link": "N/A",
                    "price": float("inf"),
                    "temperature": "N/A"
                }

            cheapest = min(
                products,
                key=lambda p: p.get("price"),
            )
            return {
                "platform": platform_name,
                "title": cheapest.get("title"),
                "link": cheapest.get("link"),
                "price": cheapest.get("price"),
                "temperature": cheapest.get("temperature", "N/A")
            }

        platform_cheapest = {}
        for platform_id, platform_config in PLATFORMS.items():
            if self.enabled_platforms.get(platform_id, True):
                products = platform_results.get(platform_id, [])

                platform_cheapest[platform_id] = get_cheapest(
                    products, platform_config["name"]
                )

        all_cheapest = list(platform_cheapest.values())
        overall_cheapest = (
            min(all_cheapest, key=lambda p: p["price"]) if all_cheapest else None
        )

        return {
            "platform_cheapest": platform_cheapest,
            "overall_cheapest": overall_cheapest,
            "platform_results": platform_results,
        }


class PriceComparisonGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Price Comparison Tool")
        self.setMinimumSize(900, 700)

        self.scraper_thread = None
        self.db_manager = DatabaseManager()

        self.init_ui()
        self.load_preferences()

    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("🛒 Price Comparison Tool")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        # Tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Search Tab
        search_tab = QWidget()
        self.init_search_tab(search_tab)
        self.tabs.addTab(search_tab, "New Search")

        # History Tab
        history_tab = QWidget()
        self.init_history_tab(history_tab)
        self.tabs.addTab(history_tab, "Search History")

    def init_search_tab(self, parent):
        """Init the search tab"""
        layout = QVBoxLayout(parent)
        layout.setSpacing(15)

        # Product Name Input
        input_group = QGroupBox("Search Settings")
        input_layout = QVBoxLayout()

        product_layout = QHBoxLayout()
        product_label = QLabel("Product Name:")
        product_label.setMinimumWidth(120)
        self.product_input = QLineEdit()
        self.product_input.setPlaceholderText("e.g., Apple iPhone 17 Pro 256GB")
        product_layout.addWidget(product_label)
        product_layout.addWidget(self.product_input)
        input_layout.addLayout(product_layout)

        # Similarity Threshold
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Similarity Threshold:")
        threshold_label.setMinimumWidth(120)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setMinimum(0.0)
        self.threshold_spin.setMaximum(1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(0.85)
        self.threshold_spin.setToolTip(
            "Minimum similarity score (0-1) to keep a product"
        )
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_spin)
        threshold_layout.addStretch()
        input_layout.addLayout(threshold_layout)

        # User Data Directory
        data_dir_layout = QHBoxLayout()
        data_dir_label = QLabel("Chrome Profile Dir:")
        data_dir_label.setMinimumWidth(120)
        self.data_dir_input = QLineEdit()
        self.data_dir_input.setToolTip("Path to Chrome user data directory")
        self.data_dir_input.textChanged.connect(self.save_preferences)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_data_dir)
        data_dir_layout.addWidget(data_dir_label)
        data_dir_layout.addWidget(self.data_dir_input)
        data_dir_layout.addWidget(browse_btn)
        input_layout.addLayout(data_dir_layout)

        headless_layout = QHBoxLayout()
        self.headless_checkbox = QCheckBox("Run in headless mode (no browser window)")
        headless_layout.addWidget(self.headless_checkbox)
        headless_layout.addStretch()
        input_layout.addLayout(headless_layout)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Platform Selection
        platform_group = QGroupBox("Select Platforms")
        platform_layout = QHBoxLayout()

        self.platform_checkboxes = {}
        for platform_id, platform_config in PLATFORMS.items():
            checkbox = QCheckBox(platform_config["name"])
            checkbox.setChecked(platform_config["enabled"])
            self.platform_checkboxes[platform_id] = checkbox
            platform_layout.addWidget(checkbox)

        platform_group.setLayout(platform_layout)
        layout.addWidget(platform_group)

        # Control Buttons
        self.start_btn = QPushButton("🔍 Start Search")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self.start_search)
        layout.addWidget(self.start_btn)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        # Results Display
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(300)
        results_layout.addWidget(self.results_text)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

    def init_history_tab(self, parent):
        """Initialize the history tab"""
        layout = QVBoxLayout(parent)
        layout.setSpacing(15)

        # Searches Table
        searches_group = QGroupBox("Previous Searches")
        searches_layout = QVBoxLayout()

        # Build column headers dynamically based on platforms
        base_headers = ["ID", "Product", "Threshold", "Date"]
        platform_headers = []
        for platform_id, platform_config in PLATFORMS.items():
            platform_name = platform_config["name"]
            platform_headers.extend([f"{platform_name} Link", f"{platform_name} Price", f"{platform_name} Hot"])
        final_headers = base_headers + platform_headers + ["Cheapest Price"]

        self.searches_table = QTableWidget()
        self.searches_table.setColumnCount(len(final_headers))
        self.searches_table.setHorizontalHeaderLabels(final_headers)

        # Enable column reordering
        header = self.searches_table.horizontalHeader()
        header.setSectionsMovable(True)
        header.setDragEnabled(True)
        header.setDragDropMode(QHeaderView.DragDropMode.InternalMove)

        # Save column order when sections are moved
        header.sectionMoved.connect(self.save_column_order)

        # Set stretch for certain columns
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # Product Name

        self.searches_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.searches_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.searches_table.itemSelectionChanged.connect(self.on_search_selected)

        searches_layout.addWidget(self.searches_table)

        # Buttons for searches
        search_buttons = QHBoxLayout()

        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.load_searches)
        search_buttons.addWidget(self.refresh_btn)

        self.delete_search_btn = QPushButton("🗑️ Delete Search")
        self.delete_search_btn.clicked.connect(self.delete_selected_search)
        self.delete_search_btn.setEnabled(False)
        search_buttons.addWidget(self.delete_search_btn)

        self.export_search_btn = QPushButton("💾 Export Selected")
        self.export_search_btn.clicked.connect(self.export_selected_search)
        self.export_search_btn.setEnabled(False)
        search_buttons.addWidget(self.export_search_btn)

        self.export_all_btn = QPushButton("💾 Export All")
        self.export_all_btn.clicked.connect(self.export_all_searches)
        search_buttons.addWidget(self.export_all_btn)

        self.reset_columns_btn = QPushButton("↺ Reset Columns")
        self.reset_columns_btn.clicked.connect(self.reset_column_order)
        search_buttons.addWidget(self.reset_columns_btn)

        searches_layout.addLayout(search_buttons)
        searches_group.setLayout(searches_layout)
        layout.addWidget(searches_group)

        # Apply saved column order after table is created
        self.load_and_apply_column_order()

    def load_preferences(self):
        """Load saved preferences from database"""
        # Load user data directory
        user_data_dir = self.db_manager.get_preference("user_data_dir", "")
        self.data_dir_input.setText(user_data_dir)

    def load_and_apply_column_order(self):
        """Load and apply saved column order to the table"""
        column_order_json = self.db_manager.get_preference("column_order")
        if column_order_json:
            try:
                column_order = json.loads(column_order_json)
                self.apply_column_order(column_order)
            except json.JSONDecodeError:
                pass

    def save_preferences(self):
        """Save preferences to database"""
        self.db_manager.save_preference("user_data_dir", self.data_dir_input.text())

    def save_column_order(self):
        """Save the current column order to database"""
        header = self.searches_table.horizontalHeader()
        column_count = header.count()

        # Get visual index for each logical index
        column_order = []
        for logical_idx in range(column_count):
            visual_idx = header.visualIndex(logical_idx)
            column_order.append(visual_idx)

        # Save to database
        self.db_manager.save_preference("column_order", json.dumps(column_order))

    def apply_column_order(self, column_order: List[int]):
        """Apply saved column order to table"""
        header = self.searches_table.horizontalHeader()
        column_count = header.count()

        # Validate column order
        if len(column_order) != column_count:
            print(
                f"Warning: Saved column order length ({len(column_order)}) doesn't match current column count ({column_count}). Ignoring saved order."
            )
            return

        # Validate all indices are valid
        if not all(0 <= idx < column_count for idx in column_order):
            print(
                "Warning: Invalid column indices in saved order. Ignoring saved order."
            )
            return

        # Create a mapping of logical index to target visual index
        # column_order[logical_idx] = target_visual_idx
        for logical_idx in range(column_count):
            target_visual_idx = column_order[logical_idx]
            current_visual_idx = header.visualIndex(logical_idx)

            if current_visual_idx != target_visual_idx:
                # Find which logical index is currently at target_visual_idx
                for check_logical in range(column_count):
                    if header.visualIndex(check_logical) == target_visual_idx:
                        # Swap them
                        header.moveSection(current_visual_idx, target_visual_idx)
                        break

    def reset_column_order(self):
        """Reset columns to default order"""
        header = self.searches_table.horizontalHeader()
        column_count = header.count()

        # Reset to default order (logical == visual)
        for i in range(column_count):
            header.moveSection(header.visualIndex(i), i)

        # Save the reset order
        self.save_column_order()

        QMessageBox.information(self, "Success", "Column order reset to default!")

    def get_column_visual_order(self) -> tuple:
        """
        Get the current visual order of columns.
        Returns: (visual_to_logical, headers_in_visual_order) where visual_to_logical maps
        visual position to logical index, and headers are in visual order
        """
        header = self.searches_table.horizontalHeader()
        column_count = header.count()

        # Build mapping: visual position -> logical index
        visual_to_logical = []
        for visual_idx in range(column_count):
            logical_idx = header.logicalIndex(visual_idx)
            visual_to_logical.append(logical_idx)

        # Get headers in visual order
        headers_in_visual_order = []
        for visual_idx in range(column_count):
            logical_idx = header.logicalIndex(visual_idx)
            headers_in_visual_order.append(
                self.searches_table.horizontalHeaderItem(logical_idx).text()
            )

        return visual_to_logical, headers_in_visual_order

    def browse_data_dir(self):
        """Open directory browser for Chrome profile"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Chrome Profile Directory", self.data_dir_input.text()
        )
        if directory:
            self.data_dir_input.setText(directory)

    def start_search(self):
        """Start the price comparison search"""
        product_name = self.product_input.text().strip()

        if not product_name:
            QMessageBox.warning(
                self, "Input Required", "Please enter a product name to search for."
            )
            return

        # Get enabled platforms
        enabled_platforms = {
            platform_id: checkbox.isChecked()
            for platform_id, checkbox in self.platform_checkboxes.items()
        }

        if not any(enabled_platforms.values()):
            QMessageBox.warning(
                self,
                "Platform Selection",
                "Please select at least one platform to search.",
            )
            return

        # Disable controls during search
        self.start_btn.setEnabled(False)
        self.product_input.setEnabled(False)
        self.threshold_spin.setEnabled(False)
        self.data_dir_input.setEnabled(False)
        self.headless_checkbox.setEnabled(False)
        for checkbox in self.platform_checkboxes.values():
            checkbox.setEnabled(False)

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

        # Clear previous results
        self.results_text.clear()
        self.results_text.append("Starting search...\n")

        # Start scraper thread
        self.scraper_thread = ScraperThread(
            product_name,
            self.threshold_spin.value(),
            enabled_platforms,
            self.data_dir_input.text(),
            self.headless_checkbox.isChecked(),
        )
        self.scraper_thread.progress_update.connect(self.update_progress)
        self.scraper_thread.finished.connect(self.on_search_finished)
        self.scraper_thread.error.connect(self.on_search_error)
        self.scraper_thread.start()

    def update_progress(self, message: str):
        """Update progress text"""
        self.results_text.append(message)
        # Auto-scroll to bottom
        self.results_text.moveCursor(QTextCursor.MoveOperation.End)

    def on_search_finished(self, results: dict):
        """Handle search completion"""
        # Hide progress bar
        self.progress_bar.setVisible(False)

        # Display results summary
        self.results_text.append("\n" + "=" * 50)
        self.results_text.append("SEARCH COMPLETED")
        self.results_text.append("=" * 50 + "\n")

        platform_cheapest = results["platform_cheapest"]
        overall_cheapest = results["overall_cheapest"]

        # Display cheapest from each platform
        self.results_text.append("Cheapest prices by platform:")
        for platform_id, data in platform_cheapest.items():
            if data["price"] != float("inf"):
                self.results_text.append(
                    f"\n{data['platform']}: {data['price']:.2f} PLN"
                )
                self.results_text.append(f"  {data['title']}")
            else:
                self.results_text.append(f"\n{data['platform']}: No results found")

        # Display overall winner
        if overall_cheapest and overall_cheapest["price"] != float("inf"):
            self.results_text.append("\n" + "=" * 50)
            self.results_text.append(
                f"🏆 OVERALL CHEAPEST: {overall_cheapest['platform']} at {overall_cheapest['price']:.2f} PLN"
            )
            self.results_text.append("=" * 50)

        # Save to database
        try:
            search_id = self.db_manager.save_search(
                self.product_input.text(),
                self.threshold_spin.value(),
                platform_cheapest,
                overall_cheapest,
            )
            self.results_text.append(
                f"\n✅ Results saved to database (Search ID: {search_id})"
            )

            # Refresh history table
            self.load_searches()

            # Switch to history tab
            QMessageBox.information(
                self,
                "Search Complete",
                "Search completed!\nSwitch to 'Search History' tab to view and export results.",
            )
        except Exception as e:
            QMessageBox.warning(
                self,
                "Database Error",
                f"Results found but failed to save to database:\n{str(e)}",
            )

        # Re-enable controls
        self.start_btn.setEnabled(True)
        self.product_input.setEnabled(True)
        self.threshold_spin.setEnabled(True)
        self.data_dir_input.setEnabled(True)
        self.headless_checkbox.setEnabled(True)
        for checkbox in self.platform_checkboxes.values():
            checkbox.setEnabled(True)

    def on_search_error(self, error_message: str):
        """Handle search errors"""
        self.progress_bar.setVisible(False)

        QMessageBox.critical(
            self,
            "Search Error",
            f"An error occurred during the search:\n\n{error_message}",
        )

        self.results_text.append(f"\n❌ ERROR: {error_message}")

        # Re-enable controls
        self.start_btn.setEnabled(True)
        self.product_input.setEnabled(True)
        self.threshold_spin.setEnabled(True)
        self.data_dir_input.setEnabled(True)
        self.headless_checkbox.setEnabled(True)
        for checkbox in self.platform_checkboxes.values():
            checkbox.setEnabled(True)

    def load_searches(self):
        """Load all searches from database"""
        searches = self.db_manager.get_all_searches_with_platforms()

        self.searches_table.setRowCount(0)

        for search in searches:
            row = self.searches_table.rowCount()
            self.searches_table.insertRow(row)

            col = 0

            # ID
            self.searches_table.setItem(row, col, QTableWidgetItem(str(search["id"])))
            col += 1

            # Product Name
            self.searches_table.setItem(
                row, col, QTableWidgetItem(search["product_name"])
            )
            col += 1

            # Threshold
            self.searches_table.setItem(
                row, col, QTableWidgetItem(f"{search['threshold']:.2f}")
            )
            col += 1

            # Date
            self.searches_table.setItem(row, col, QTableWidgetItem(search["date"]))
            col += 1

            # Platform columns (Link and Price for each platform)
            for platform_id, platform_config in PLATFORMS.items():
                platform_name = platform_config["name"]
                platform_data = search["platforms"].get(platform_name, {})

                # Link
                link = platform_data.get("link", "N/A")
                link_item = QTableWidgetItem(link if link != "N/A" else "N/A")
                self.searches_table.setItem(row, col, link_item)
                col += 1

                # Price
                price = platform_data.get("price")
                if price is not None:
                    price_text = f"{price:.2f}"
                    price_item = QTableWidgetItem(price_text)

                else:
                    price_item = QTableWidgetItem("N/A")

                self.searches_table.setItem(row, col, price_item)
                col += 1

                # temperature
                temperature = platform_data.get("temperature")
                if temperature is not None:
                    temperature_item = QTableWidgetItem(temperature)

                else:
                    temperature_item = QTableWidgetItem("N/A")

                self.searches_table.setItem(row, col, temperature_item)
                col += 1

            # Overall Cheapest Price
            overall_price = search["overall_price"]
            if overall_price:
                overall_item = QTableWidgetItem(f"{overall_price:.2f}")
            else:
                overall_item = QTableWidgetItem("N/A")

            self.searches_table.setItem(row, col, overall_item)

    def on_search_selected(self):
        """Handle search selection"""
        selected_rows = self.searches_table.selectionModel().selectedRows()

        if selected_rows:
            self.delete_search_btn.setEnabled(True)
            self.export_search_btn.setEnabled(True)

        else:
            self.delete_search_btn.setEnabled(False)
            self.export_search_btn.setEnabled(False)

    def delete_selected_search(self):
        """Delete the selected search"""
        selected_rows = self.searches_table.selectionModel().selectedRows()

        if not selected_rows:
            return

        row = selected_rows[0].row()
        search_id = int(self.searches_table.item(row, 0).text())
        product_name = self.searches_table.item(row, 1).text()

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete the search for '{product_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.db_manager.delete_search(search_id)
                self.load_searches()
                QMessageBox.information(self, "Success", "Search deleted successfully!")
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to delete search:\n{str(e)}"
                )

    def export_selected_search(self):
        """Export the selected search to CSV matching table format"""
        selected_rows = self.searches_table.selectionModel().selectedRows()

        if not selected_rows:
            return

        row = selected_rows[0].row()
        search_id = int(self.searches_table.item(row, 0).text())
        product_name = self.searches_table.item(row, 1).text()

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Search to CSV",
            f"search_{search_id}_{product_name.replace(' ', '_')}.csv",
            "CSV Files (*.csv)",
        )

        if file_path:
            try:
                # Get current column order
                visual_to_logical, headers_in_visual_order = (
                    self.get_column_visual_order()
                )

                self.db_manager.export_searches_with_column_order(
                    file_path,
                    search_ids=[search_id],
                    visual_to_logical=visual_to_logical,
                    headers_in_visual_order=headers_in_visual_order,
                )
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Search exported successfully to:\n{file_path}\n\nExported with current column order!",
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Error", f"Failed to export search:\n{str(e)}"
                )

    def export_all_searches(self):
        """Export all searches to CSV matching table format"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export All Searches to CSV", "all_searches.csv", "CSV Files (*.csv)"
        )

        if file_path:
            try:
                # Get current column order
                visual_to_logical, headers_in_visual_order = (
                    self.get_column_visual_order()
                )

                self.db_manager.export_searches_with_column_order(
                    file_path,
                    visual_to_logical=visual_to_logical,
                    headers_in_visual_order=headers_in_visual_order,
                )
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"All searches exported successfully to:\n{file_path}\n\nExported with current column order!",
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Error", f"Failed to export searches:\n{str(e)}"
                )


def main():
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    window = PriceComparisonGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
