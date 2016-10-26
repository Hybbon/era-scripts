from pureSVD import PureSVDPredictor
from predictor import Predictor
import argparse
from operator import itemgetter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("base", type=str)
    parser.add_argument("out", type=str)
    parser.add_argument("-s", "--separator", type=str, default="\t")
    parser.add_argument("-f", "--factors", type=int, default=50)
    parser.add_argument("-n", "--num_items", type=int, default=100)

    return parser.parse_args()


def generate_rankings(recommender, users, items, num_items):
    item_ids = items.keys()
    rankings = {}
    for user, user_rel in users.items():
        u_item_ids = list(set(item_ids) - set(user_rel.keys()))
        all_scores = recommender.get_ratings(user, u_item_ids)
        rankings[user] = sorted(all_scores, reverse=True)[:num_items]
    return rankings


def save_rankings(path, rankings):
    f = open(path, "w")
    sorted_rankings = sorted(rankings.items(), key=itemgetter(0))
    for user, ranking in sorted_rankings:
        f.write("%s\t" % user)
        l = ["%s:%s" % (item, score) for score, item in ranking]
        f.write("[%s]\n" % ",".join(l))
    f.close()


def main():
    args = parse_args()

    training = Predictor(args.base, args.separator)
    users, items = training.store_data_relations()

    recommender = PureSVDPredictor(items, users, args.factors)

    rankings = generate_rankings(recommender, users, items, args.num_items)

    save_rankings(args.out, rankings)


if __name__ == "__main__":
    main()
