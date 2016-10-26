import re

ratings_regex_str = "(?P<user_id>\d+)\s+(?P<item_id>\d+)\s+(?:\d+)\s*"
ratings_regex = re.compile(ratings_regex_str)


def ratings_dict(path):
    """Generates a dictionary from a list of user ratings.

    path -- path to the ratings file."""
    f = open(path, 'r')
    d = {}
    prev_user = None
    for line in f:
        result = ratings_regex.match(line)
        user_id = int(result.group('user_id'))
        item_id = int(result.group('item_id'))
        if user_id != prev_user:
            if prev_user:
                d[prev_user] = items
            prev_user = user_id
            items = [item_id]
        else:
            items.append(item_id)
    f.close()
    return d


results_user_id_regex_str = "(\d+)"
results_user_id_regex = re.compile(results_user_id_regex_str)

float_regex = r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
results_items_regex_str = "(\d+):({0})".format(float_regex)
results_items_regex = re.compile(results_items_regex_str)


def rankings_dict(path):
    """Generates a dictionary from a file of ranking algorithm results.

    path -- path to the results file."""
    f = open(path, 'r')
    d = {}
    for line in f:
        user_id_result = results_user_id_regex.match(line)
        user_id = int(user_id_result.group(0))
        ranking = results_items_regex.findall(line, user_id_result.end())
        # Assuming results are already sorted in descending order
        items = [int(i[0]) for i in ranking]
        d[user_id] = items
    f.close()
    return d
