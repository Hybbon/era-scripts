#!/usr/bin/env python3

"""
filter_yelp.py
==============
Filters the yelp dataset to a single location
---------------------------------------------

This script has a very simple and specific use case: filtering the
Yelp Challenge dataset to a single specified location. It can be
called in two different ways. All of them require a single positional
argument, which is the path to the folder with the Yelp Dataset. More
specifically, two files are required:

- yelp_academic_dataset_review.json
- yelp_academic_dataset_business.json

These can be downloaded at the `Yelp Dataset Challenge Website`_.

.. _Yelp Dataset Challenge Website: https://www.yelp.com/dataset_challenge

Firstly, the script can be called without any extra parameters. It
will, then, output ``yelp_script_out.json`` at the data folder with
only reviews for the city of Pittsburgh. To specify a location, set
it via ``-c <location>`` flag. To see all possible locations, use the
``-l`` flag. Examples below::

    python filter_yelp.py yelp_dataset/ -l
    python filter_yelp.py yelp_dataset/ -c "Multi-word location in quotes"
"""

import argparse
import os.path
import re
import ipdb

BUSINESS_FILENAME = "yelp_academic_dataset_business.json"
REVIEW_FILENAME = "yelp_academic_dataset_review.json"
USERS_FILENAME = "yelp_academic_dataset_user.json"


def parse_args():
    """Parse command line parameters"""
    p = argparse.ArgumentParser()
    p.add_argument("yelp", help="Path to the yelp dataset folder")
    p.add_argument("-o", "--out", default="yelp_script_out.json",
            help="Name of the output file")
    p.add_argument("--list_cities", "-l", action="store_true",
            help="List the available cities in the dataset and do nothing")
    p.add_argument("--city", "-c", default="Pittsburgh",
            help="Name of the city to be used")
    return p.parse_args()


def tag_content(json, tag):
    needle = '"{tag}":"'.format(tag=tag)
    #ipdb.set_trace()
    json = json.replace(" ","")
    match = json.find(needle)
    if match == -1:
        return None
    begin = match + len(needle)
    end = json.find('"', begin)
    return json[begin:end]


def list_cities(business_path):
    cities = {}#set()
    for line_no, business in enumerate(open(business_path), start=1):
        match = tag_content(business, "city")
        if match:
            if match in cities:
                cities[match] += 1
            else:
                cities[match] = 1
            #cities.add(match)
        else:
            print("WARNING: Business in line {} has no city".format(line_no))
    return cities


def businesses_in_city(business_path, city):
    businesses = []
    for line_no, business in enumerate(open(business_path), start=1):
        match = tag_content(business, "city")
        #ipdb.set_trace()
        if match:
            if match == city:
                business_id = tag_content(business, "business_id")
                businesses.append(business_id)
        else:
            print("WARNING: Business in line {} has no city".format(line_no))
    return set(businesses)


def filter_reviews(business_path, review_path, out_path, city):
    city_businesses = businesses_in_city(business_path, city)
    print("{} businesses in {}".format(len(city_businesses), city))
    with open(review_path) as reviews, open(out_path, "w") as out:
        for line_no, review in enumerate(reviews, start=1):
            review_business = tag_content(review, "business_id")
            if review_business:
                if review_business in city_businesses:
                    out.write(review)
            else:
                print("WARNING: Review in line {} has no "
                      "business".format(line_no))


def main():
    args = parse_args()
    business_path = os.path.join(args.yelp, BUSINESS_FILENAME)
    if args.list_cities:
        cities = list_cities(business_path)
        sorted_cities = sorted(list(cities.items()),key=lambda tup : tup[1],reverse=True)
        for city in sorted_cities[:20]:
            print(city)
    else:
        out_path = os.path.join(args.yelp, args.out)
        review_path = os.path.join(args.yelp, REVIEW_FILENAME)
        filter_reviews(business_path, review_path, out_path, args.city)


if __name__ == "__main__":
    main()
