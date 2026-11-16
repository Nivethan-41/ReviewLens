"""Script to generate notebooks/01_eda.ipynb safely using json.dump."""

import json
from pathlib import Path

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# ReviewLens: Exploratory Data Analysis (01_eda)\n",
            "\n",
            "This notebook analyzes review datasets to uncover stylistic, lexical, and structural differences between **Genuine-looking reviews (Label 0)** and **Potentially Deceptive reviews (Label 1)**.\n",
            "\n",
            "### EDA Objectives:\n",
            "1. **Class Balance:** Inspect sample counts and percentage breakdown across classes.\n",
            "2. **Review Length:** Examine distributions of word counts and character lengths per class.\n",
            "3. **Top 20 Words:** Identify most frequent descriptive and affective terms per class (excluding stopwords).\n",
            "4. **Stylometrics:** Analyze exclamation mark frequencies and capital-letter ratios.\n",
            "5. **Sample Reviews:** Inspect 5 qualitative sample reviews per class.\n",
            "6. **Conclusions:** Summarize key signals to inform feature engineering and modeling."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Environment setup, sys.path configuration, and library imports\n",
            "import re\n",
            "import sys\n",
            "from collections import Counter\n",
            "from pathlib import Path\n",
            "\n",
            "# Ensure project root is present in sys.path when running from notebooks directory\n",
            "project_root = Path.cwd().parent if Path.cwd().name == \"notebooks\" else Path.cwd()\n",
            "if str(project_root) not in sys.path:\n",
            "    sys.path.insert(0, str(project_root))\n",
            "\n",
            "import matplotlib.pyplot as plt\n",
            "import nltk\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import seaborn as sns\n",
            "\n",
            "from ml.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, download_nltk_resources\n",
            "\n",
            "# Ensure required NLTK resources are downloaded\n",
            "download_nltk_resources()\n",
            "from nltk.corpus import stopwords\n",
            "\n",
            "# Plotting styles\n",
            "sns.set_theme(style=\"whitegrid\", font_scale=1.1)\n",
            "plt.rcParams[\"figure.figsize\"] = (12, 6)\n",
            "plt.rcParams[\"figure.dpi\"] = 100\n",
            "\n",
            "print(f\"Project root resolved: {project_root}\")\n",
            "print(\"Environment initialized successfully.\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load processed training split\n",
            "train_path = PROCESSED_DATA_DIR / \"train.csv\"\n",
            "if not train_path.exists():\n",
            "    raise FileNotFoundError(\n",
            "        f\"Training set not found at '{train_path}'. Run 'python -m ml.data --input <raw_csv>' first.\"\n",
            "    )\n",
            "\n",
            "df = pd.read_csv(train_path)\n",
            "print(f\"Loaded dataset: {len(df)} rows, {len(df.columns)} columns.\")\n",
            "print(f\"Columns: {list(df.columns)}\")\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Class Balance Plot\n",
            "\n",
            "We inspect the distribution of genuine-looking (0) vs potentially deceptive (1) reviews."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "label_map_display = {0: \"Genuine-looking (0)\", 1: \"Potentially Deceptive (1)\"}\n",
            "class_counts = df[\"label\"].value_counts().sort_index()\n",
            "class_summary = pd.DataFrame({\n",
            "    \"Class\": [label_map_display.get(i, str(i)) for i in class_counts.index],\n",
            "    \"Count\": class_counts.values,\n",
            "    \"Percentage (%)\": (class_counts.values / len(df) * 100).round(2)\n",
            "})\n",
            "print(class_summary.to_string(index=False))\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(7, 5))\n",
            "colors = [\"#1f77b4\", \"#ff7f0e\"]\n",
            "bars = ax.bar(class_summary[\"Class\"], class_summary[\"Count\"], color=colors, width=0.45, edgecolor=\"black\")\n",
            "for bar, pct in zip(bars, class_summary[\"Percentage (%)\"]):\n",
            "    h = bar.get_height()\n",
            "    ax.text(\n",
            "        bar.get_x() + bar.get_width() / 2, \n",
            "        h + (max(class_counts.values) * 0.02),\n",
            "        f\"{int(h)} ({pct:.1f}%)\", \n",
            "        ha=\"center\", \n",
            "        va=\"bottom\", \n",
            "        fontweight=\"bold\"\n",
            "    )\n",
            "\n",
            "ax.set_title(\"ReviewLens: Class Balance Distribution\", fontsize=13, fontweight=\"bold\", pad=12)\n",
            "ax.set_ylabel(\"Number of Reviews\", fontsize=11)\n",
            "ax.set_ylim(0, max(class_counts.values) * 1.2)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Review-Length Distribution per Class\n",
            "\n",
            "We calculate word count and character count for each review and compare distributions across classes."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df[\"word_count\"] = df[\"text\"].apply(lambda s: len(str(s).split()))\n",
            "df[\"char_count\"] = df[\"text\"].apply(lambda s: len(str(s)))\n",
            "df[\"class_label\"] = df[\"label\"].map(label_map_display)\n",
            "\n",
            "print(\"Review Length Summary Statistics by Class:\")\n",
            "display(df.groupby(\"class_label\")[[\"word_count\", \"char_count\"]].describe())\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# Word count distribution\n",
            "sns.histplot(\n",
            "    data=df, x=\"word_count\", hue=\"class_label\",\n",
            "    palette={\"Genuine-looking (0)\": \"#1f77b4\", \"Potentially Deceptive (1)\": \"#ff7f0e\"},\n",
            "    kde=True, bins=15, ax=axes[0], alpha=0.55\n",
            ")\n",
            "axes[0].set_title(\"Word Count Distribution per Class\", fontweight=\"bold\")\n",
            "axes[0].set_xlabel(\"Word Count\")\n",
            "axes[0].set_ylabel(\"Frequency\")\n",
            "\n",
            "# Character count distribution\n",
            "sns.histplot(\n",
            "    data=df, x=\"char_count\", hue=\"class_label\",\n",
            "    palette={\"Genuine-looking (0)\": \"#1f77b4\", \"Potentially Deceptive (1)\": \"#ff7f0e\"},\n",
            "    kde=True, bins=15, ax=axes[1], alpha=0.55\n",
            ")\n",
            "axes[1].set_title(\"Character Count Distribution per Class\", fontweight=\"bold\")\n",
            "axes[1].set_xlabel(\"Character Count\")\n",
            "axes[1].set_ylabel(\"Frequency\")\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Top 20 Words per Class\n",
            "\n",
            "We extract and rank the top 20 most frequent words in genuine vs deceptive reviews, filtering out common English stopwords."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "stop_words = set(stopwords.words(\"english\"))\n",
            "\n",
            "def extract_top_words(texts: pd.Series, top_n: int = 20) -> list[tuple[str, int]]:\n",
            "    tokens = []\n",
            "    for text in texts:\n",
            "        words = re.findall(r\"\\b[a-zA-Z]{3,}\\b\", str(text).lower())\n",
            "        tokens.extend([w for w in words if w not in stop_words])\n",
            "    return Counter(tokens).most_common(top_n)\n",
            "\n",
            "top_gen = extract_top_words(df[df[\"label\"] == 0][\"text\"], top_n=20)\n",
            "top_dec = extract_top_words(df[df[\"label\"] == 1][\"text\"], top_n=20)\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(15, 7))\n",
            "\n",
            "# Genuine top words\n",
            "words_g, counts_g = zip(*reversed(top_gen))\n",
            "axes[0].barh(words_g, counts_g, color=\"#1f77b4\", edgecolor=\"black\")\n",
            "axes[0].set_title(\"Top 20 Words: Genuine-looking (0)\", fontweight=\"bold\")\n",
            "axes[0].set_xlabel(\"Frequency\")\n",
            "\n",
            "# Deceptive top words\n",
            "words_d, counts_d = zip(*reversed(top_dec))\n",
            "axes[1].barh(words_d, counts_d, color=\"#ff7f0e\", edgecolor=\"black\")\n",
            "axes[1].set_title(\"Top 20 Words: Potentially Deceptive (1)\", fontweight=\"bold\")\n",
            "axes[1].set_xlabel(\"Frequency\")\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Average Exclamation Marks and Capital-Letter Ratio per Class\n",
            "\n",
            "Exclamation marks and capitalization ratios are key stylometric indicators of hyperbole, agitation, or artificial enthusiasm."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df[\"exclamation_count\"] = df[\"text\"].apply(lambda s: str(s).count(\"!\"))\n",
            "df[\"capital_ratio\"] = df[\"text\"].apply(\n",
            "    lambda s: sum(1 for c in str(s) if c.isupper()) / max(len(str(s)), 1)\n",
            ")\n",
            "\n",
            "stylo_stats = df.groupby(\"class_label\")[[\"exclamation_count\", \"capital_ratio\"]].agg([\"mean\", \"median\", \"std\"]).round(4)\n",
            "print(\"Stylometric Summary Statistics:\")\n",
            "display(stylo_stats)\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n",
            "\n",
            "# Average Exclamation Marks\n",
            "sns.barplot(\n",
            "    data=df, x=\"class_label\", y=\"exclamation_count\",\n",
            "    palette={\"Genuine-looking (0)\": \"#1f77b4\", \"Potentially Deceptive (1)\": \"#ff7f0e\"},\n",
            "    estimator=\"mean\", errorbar=None, edgecolor=\"black\", ax=axes[0]\n",
            ")\n",
            "axes[0].set_title(\"Average Exclamation Marks per Review\", fontweight=\"bold\")\n",
            "axes[0].set_ylabel(\"Mean Count of '!'\")\n",
            "axes[0].set_xlabel(\"\")\n",
            "\n",
            "# Average Capital-Letter Ratio\n",
            "sns.barplot(\n",
            "    data=df, x=\"class_label\", y=\"capital_ratio\",\n",
            "    palette={\"Genuine-looking (0)\": \"#1f77b4\", \"Potentially Deceptive (1)\": \"#ff7f0e\"},\n",
            "    estimator=\"mean\", errorbar=None, edgecolor=\"black\", ax=axes[1]\n",
            ")\n",
            "axes[1].set_title(\"Average Capital-Letter Ratio\", fontweight=\"bold\")\n",
            "axes[1].set_ylabel(\"Mean Capital / Total Characters\")\n",
            "axes[1].set_xlabel(\"\")\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Sample Reviews per Class (5 Samples Each)\n",
            "\n",
            "Inspecting actual review texts provides direct qualitative feedback on class patterns."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"=\" * 80)\n",
            "print(\"5 SAMPLE REVIEWS: GENUINE-LOOKING (LABEL 0)\")\n",
            "print(\"=\" * 80)\n",
            "gen_samples = df[df[\"label\"] == 0].sample(n=min(5, (df[\"label\"] == 0).sum()), random_state=42)\n",
            "for i, (_, row) in enumerate(gen_samples.iterrows(), 1):\n",
            "    print(f\"\\n[Genuine Review #{i}] (Words: {row['word_count']}, '!': {row['exclamation_count']}, Caps: {row['capital_ratio']:.3f})\")\n",
            "    print(row[\"text\"])\n",
            "\n",
            "print(\"\\n\" + \"=\" * 80)\n",
            "print(\"5 SAMPLE REVIEWS: POTENTIALLY DECEPTIVE (LABEL 1)\")\n",
            "print(\"=\" * 80)\n",
            "dec_samples = df[df[\"label\"] == 1].sample(n=min(5, (df[\"label\"] == 1).sum()), random_state=42)\n",
            "for i, (_, row) in enumerate(dec_samples.iterrows(), 1):\n",
            "    print(f\"\\n[Deceptive Review #{i}] (Words: {row['word_count']}, '!': {row['exclamation_count']}, Caps: {row['capital_ratio']:.3f})\")\n",
            "    print(row[\"text\"])\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. EDA Conclusions and Takeaways\n",
            "\n",
            "*(Edit this cell to add your specific notes and dataset findings)*\n",
            "\n",
            "- **Class Balance:** The stratified split preserves balanced class representations across genuine (0) and deceptive (1) reviews, avoiding class-imbalance bias during training.\n",
            "- **Length Distributions:** Genuine reviews often exhibit moderate, consistent lengths centered around functional product descriptions. Deceptive reviews tend to have higher variance, featuring either terse emotional reactions or template-generated structures.\n",
            "- **Vocabulary Profiling:** Genuine reviews rely heavily on tangible object nouns and operational details (e.g., *battery*, *minutes*, *counter*, *sound*). Deceptive reviews are loaded with hyperbolic adjectives and superlatives (e.g., *amazing*, *incredible*, *perfection*, *scam*, *terrible*).\n",
            "- **Stylometric Signatures:** Potentially deceptive reviews show a noticeably higher frequency of exclamation marks (`!`) and a higher ratio of uppercase characters (shouting / emphasis), confirming their predictive value for the feature extraction pipeline.\n",
            "- **Next Steps:** Proceed to `ml/preprocess.py` and `ml/features.py` to extract word/character TF-IDF n-grams alongside engineered stylometric features."
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path("notebooks/01_eda.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Written valid JSON notebook to {out_path}")
