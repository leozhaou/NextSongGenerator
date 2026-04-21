import csv
import random
import re



def language_filter(file, language = "en"):
    """
    input:
        file - csv of data
        language - language to filter by (in our case english)

    output:
        file - csv of data with english
    """

    with open (file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)

        # getting index of language columns
        lang_idx1 = header.index("language_cld3")
        lang_idx2 = header.index("language_ft")

        rows = [row for row in reader if row[lang_idx1] == "en" and row[lang_idx2] == "en"]

        with open (file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)


def random_sample(file, fraction=1/8, seed=42):
    random.seed(seed)
    with open(file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    sampled = random.sample(rows, k=int(len(rows) * fraction))

    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(sampled)


def preprocess_lyrics(file):
    with open(file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    lyrics_idx = header.index("lyrics")

    for row in rows:
        print("I am a big dummy")
        lyrics = row[lyrics_idx]
        lyrics = lyrics.lower()
        lyrics = re.sub(r"\[.*?\]", "", lyrics)
        lyrics = re.sub(r"[^a-z0-9\s']", "", lyrics)
        lyrics = re.sub(r"\s+", " ", lyrics).strip()
        row[lyrics_idx] = lyrics

    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


language_filter("song_lyrics.csv")
random_sample("song_lyrics.csv")
preprocess_lyrics("song_lyrics.csv")

        


