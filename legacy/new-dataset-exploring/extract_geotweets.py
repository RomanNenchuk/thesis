#!/usr/bin/env python3
"""
Stream a monthly Archive Team "Twitter Stream Grab" tar and keep only tweets that carry
precise GPS coordinates (coordinates.type == "Point").

The tar contains many small *.json.bz2 files (one tweet JSON object per line). The tar is
read in streaming mode, so nothing is unpacked to disk and a truncated tar (for example the
first few GB fetched with `curl -r`) is fine: extraction simply stops at the end of the data.

Usage examples
    # local (possibly partial) tar
    python extract_geotweets.py --source part.tar --out geo_2014_06.csv

    # stream straight from archive.org, stop after ~4 GB
    python extract_geotweets.py \
        --source https://archive.org/download/archiveteam-twitter-stream-2014-06/archiveteam-twitter-stream-2014-06.tar \
        --max-gb 4 --out geo_2014_06.csv

Output columns
    tweet_id, created_at, lang, lon, lat, user_hash, source_app, place_country, text

Privacy: user ids are replaced by a salted SHA-256 prefix; raw tweets should stay local and
must not be committed or published (publish aggregates only).
"""
import argparse
import bz2
import csv
import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
import time
import urllib.request
from collections import Counter

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

FIELDS = ["tweet_id", "created_at", "lang", "lon", "lat",
          "user_hash", "source_app", "place_country", "text"]


class LimitedReader(io.RawIOBase):
    """Wrap a binary stream and report EOF after `limit` bytes (None = no limit)."""

    def __init__(self, raw, limit=None):
        self.raw = raw
        self.limit = limit
        self.read_bytes = 0

    def readable(self):
        return True

    def readinto(self, b):
        if self.limit is not None:
            remaining = self.limit - self.read_bytes
            if remaining <= 0:
                return 0
            b = memoryview(b)[:remaining]
        data = self.raw.read(len(b))
        n = len(data)
        b[:n] = data
        self.read_bytes += n
        return n


def open_source(source, max_bytes):
    if source.startswith(("http://", "https://")):
        req = urllib.request.Request(source, headers={"User-Agent": "thesis-geotweet-extractor/1.0"})
        raw = urllib.request.urlopen(req, timeout=120)
    else:
        raw = open(source, "rb")
    return LimitedReader(raw, max_bytes)


def open_member(fileobj, name):
    if name.endswith(".bz2"):
        return bz2.BZ2File(fileobj)
    if name.endswith(".gz"):
        return gzip.GzipFile(fileobj=fileobj)
    return fileobj


def hash_user(uid, salt):
    return hashlib.sha256((salt + str(uid)).encode("utf-8")).hexdigest()[:16]


def process_stream(lines, writer, stats, salt):
    """`lines` yields raw bytes lines (no TextIOWrapper: tar stream members are not seekable)."""
    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            stats["bad_json"] += 1
            continue
        if "delete" in obj or "text" not in obj:        # delete notices, limit notices, etc.
            stats["non_tweet"] += 1
            continue
        stats["tweets"] += 1
        if "retweeted_status" in obj:
            stats["retweets_skipped"] += 1
            continue
        coords = obj.get("coordinates")
        if not coords or coords.get("type") != "Point":
            if obj.get("place"):
                stats["place_only"] += 1
            continue
        try:
            lon, lat = coords["coordinates"][0], coords["coordinates"][1]   # GeoJSON order: lon, lat
        except (KeyError, IndexError, TypeError):
            stats["bad_coords"] += 1
            continue
        user = obj.get("user") or {}
        place = obj.get("place") or {}
        text = WS_RE.sub(" ", obj.get("text", "")).strip()
        writer.writerow([
            obj.get("id_str") or obj.get("id"),
            obj.get("created_at", ""),
            obj.get("lang", ""),
            lon, lat,
            hash_user(user.get("id_str") or user.get("id"), salt),
            TAG_RE.sub("", obj.get("source", "") or ""),
            place.get("country_code", ""),
            text,
        ])
        stats["geo_point"] += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="local tar path or http(s) URL of the monthly tar")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--max-gb", type=float, default=None, help="stop after reading this many GB of the tar")
    ap.add_argument("--max-files", type=int, default=None, help="stop after this many members of the tar")
    ap.add_argument("--salt", default="linkup-thesis", help="salt for hashing user ids")
    args = ap.parse_args()

    max_bytes = int(args.max_gb * 1024 ** 3) if args.max_gb else None
    src = open_source(args.source, max_bytes)
    stats = Counter()
    t0 = time.time()

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        try:
            with tarfile.open(fileobj=io.BufferedReader(src, 1 << 20), mode="r|") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    stats["files"] += 1
                    ef = tar.extractfile(member)
                    try:
                        process_stream(open_member(ef, member.name), writer, stats, args.salt)
                    except (EOFError, OSError) as e:
                        # a member cut in half (end of a partial download) or a corrupt member
                        stats["broken_members"] += 1
                        print(f"[warn] member {member.name!r}: {e}", file=sys.stderr)
                    if stats["files"] % 500 == 0:
                        mb = src.read_bytes / 1024 ** 2
                        print(f"files={stats['files']} tweets={stats['tweets']} geo_point={stats['geo_point']} "
                              f"read={mb:,.0f} MB elapsed={time.time() - t0:,.0f}s", flush=True)
                    if args.max_files and stats["files"] >= args.max_files:
                        break
        except (tarfile.ReadError, EOFError, OSError) as e:
            print(f"[info] tar stream ended early: {e} (expected for partial downloads / --max-gb)", file=sys.stderr)

    stats["bytes_read"] = src.read_bytes
    stats["seconds"] = int(time.time() - t0)
    summary = dict(stats)
    if stats["tweets"]:
        summary["geo_point_share_of_originals"] = round(stats["geo_point"] / max(stats["tweets"] - stats["retweets_skipped"], 1), 5)
    with open(args.out + ".stats.json", "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()