from .metrics import precision, ndcg


class AlgResults(object):

    """Rankings which represent recommendations of items given to users."""

    def __init__(self):
        """Initializes an empty Rankings object."""
        self.lists = {}

    @classmethod
    def from_dict(cls, d):
        alg = cls()
        for user_id, ranking in d.items():
            alg.add_user(user_id, ranking)
        return alg

    def add_user(self, user_id, ranking):
        """Adds a user's ranking to the object.

        user_id -- identifier of the user.
        ranking -- list of recommended items in descending order."""
        if type(ranking) is not list:
            ranking = list(ranking)
        self.lists[user_id] = ranking

    def intersection(self, other):
        """Creates a new Rankings object from the intersection of two objects.

        The order of the recommended items is not relevant in the intersection
        object, because that wouldn't make sense - an item could be in both
        users' rankings in different positions.

        other -- the other operand of the intersection with self."""
        new = type(self)()
        for user_id in set(self.lists.keys()) & set(other.lists.keys()):
            new.add_user(user_id, set(self.lists[user_id])
                         & set(other.lists[user_id]))
        return new

    __and__ = intersection  # Operator alias

    def union(self, other):
        """Creates a new Rankings object from the union of two objects.

        The order of the recommended items is not relevant in the union object,
        because that wouldn't make sense - an item could be in both users' ran-
        kings in different positions.

        other -- the other operand of the union with self."""
        new = type(self)()
        for user_id in set(self.lists.keys()) | set(other.lists.keys()):
            if user_id in self.lists:
                if user_id in other.lists:
                    new.add_user(user_id, set(self.lists[user_id])
                                 | set(other.lists[user_id]))
                else:
                    new.add_user(user_id, self.lists[user_id])
            else:
                new.add_user(user_id, other.lists[user_id])
        return new

    __or__ = union  # Operator alias

    def slice(self, begin, end):
        """Creates a new object which contains specific ranges of each user's
        ranking.

        [begin, end) -- range of each user's ranking which should be in the new
                        object."""
        new = type(self)()
        for user_id, user in self.lists.items():
            new.add_user(user_id, user[begin:end])
        return new

    def avg_len(self):
        """Returns the average length of the users' items lists of this
        Rankings obj."""
        return sum(len(x) for x in self.lists.values()) / len(self.lists)

    def map_metrics(self, hits, num_items=None):
        """Mean average precision of all rankings based on a hits AlgResults.

        hits -- AlgResults object which contains the hits for each user.
        num_items (optional) -- if specified, only the first num_items items of
                                each ranking shall be taken into account."""
        precisions = [0 if user_id not in hits.lists else precision(
                      ranking, hits.lists[user_id], num_items)
                      for user_id, ranking in self.lists.items()]
        # users_in_hits = [user for user in self.lists.keys() if user in hits.lists]
        # return sum(precisions) / len(users_in_hits)
        return sum(precisions) / len(precisions)

    def map_clusters(self, hits, u_cl, num_items=None):
        """Mean average precision of all rankings by cluster of users.

        hits -- AlgResults object which contains the hits for each user.
        u_cl -- dictionary which specifies each user's cluster.
        num_items (optional) -- if specified, only the first num_items items of
                                each ranking shall be taken into account."""
        clusters = [[] for cl_i in set(u_cl.values())]
        for user_id, ranking in self.lists.items():
            if user_id in u_cl:
                clusters[u_cl[user_id]].append(0 if user_id not in hits.lists
                    else precision(ranking, hits.lists[user_id], num_items))
        return [sum(cluster) / len(cluster) for cluster in clusters]

    def ndcg_metrics(self, hits, num_items=None):
        """NDCG of all rankings based on a hits AlgResults.

        hits -- AlgResults object which contains the hits for each user.
        num_items (optional) -- if specified, only the first num_items items of
                                each ranking shall be taken into account."""

        ndcgs = [0 if user_id not in hits.lists else ndcg(ranking,
                 hits.lists[user_id],num_items) for user_id, ranking
                 in self.lists.items()]
        return sum(ndcgs) / len(ndcgs)

    def ndcg_clusters(self, hits, u_cl, num_items=None):
        """NDCG of all rankings by cluster of users.

        hits -- AlgResults object which contains the hits for each user.
        u_cl -- dictionary which specifies each user's cluster.
        num_items (optional) -- if specified, only the first num_items items of
                                each ranking shall be taken into account."""

        clusters = [[] for cl_i in set(u_cl.values())]
        for user_id, ranking in self.lists.items():
            if user_id in u_cl:
                clusters[u_cl[user_id]].append(0 if user_id not in hits.lists
                    else ndcg(ranking, hits.lists[user_id], num_items))
        return [sum(cluster) / len(cluster) for cluster in clusters]
